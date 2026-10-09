"""
Stage 3 — Fine-tune BERT / RoBERTa on the weekly dataset v2.
IT22196392 — Induwara K.P.Y. | R26-IT-059

Reads the pre-tokenized student-level split from preprocess_weekly.py
(max_len 256, head+tail truncation) and trains with a class-weighted loss
(sqrt-balanced weights from class_weights.json). The best epoch is chosen on
validation macro-F1; the test set is scored once with that checkpoint.

Needs a CUDA build of PyTorch (training env: C:/Users/ASUS/.venvs/burnout-nlp-gpu).
Run:  <gpu-python> train_transformer_weekly.py --model roberta-base
      <gpu-python> train_transformer_weekly.py --model bert-base-uncased
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from preprocess_weekly import HEAD_TOKENS, ID2LABEL, LABEL2ID, OUT_DIR as DATA_DIR
from train_baseline_weekly import full_report, scores

BASE_DIR = Path(__file__).parent
SAVE_ROOT = BASE_DIR / "models" / "saved" / "weekly_v2"
PRED_DIR = BASE_DIR / "results" / "predictions"
METRICS_DIR = BASE_DIR / "results" / "metrics"
SHORT_NAME = {"bert-base-uncased": "bert", "roberta-base": "roberta"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(model_name, split):
    df = pd.read_parquet(DATA_DIR / "tokenized" / model_name / f"{split}.parquet")
    df["input_ids"] = df["input_ids"].map(lambda x: torch.tensor(x, dtype=torch.long))
    return df


def length_bucketed_batches(lengths, batch_size, shuffle, seed):
    """Batches of similar length -> far less padding (median text is ~75 tokens, max 256).
    Shuffled in chunks so each epoch still sees a different order."""
    idx = np.arange(len(lengths))
    if not shuffle:
        idx = idx[np.argsort(lengths, kind="stable")]
        return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for c in range(0, len(idx), chunk):
        part = idx[c:c + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def collate(df, rows, pad_id, device):
    ids = pad_sequence([df["input_ids"].iat[i] for i in rows], batch_first=True, padding_value=pad_id)
    return {"input_ids": ids.to(device), "attention_mask": (ids != pad_id).long().to(device)}


@torch.no_grad()
def predict(model, df, pad_id, device, batch_size=64):
    model.eval()
    lengths = df["input_ids"].map(len).to_numpy()
    logits = np.zeros((len(df), len(LABEL2ID)), dtype=np.float32)
    for rows in length_bucketed_batches(lengths, batch_size, shuffle=False, seed=0):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(**collate(df, rows, pad_id, device)).logits
        logits[rows] = out.float().cpu().numpy()
    return logits


def save_predictions(path, df, meta, logits):
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    out = meta[["student_id", "week", "status", "label_id"]].copy()
    out["pred_id"] = probs.argmax(1)
    out["pred_label"] = out["pred_id"].map(ID2LABEL)
    for i in range(len(LABEL2ID)):
        out[f"p_{ID2LABEL[i]}"] = probs[:, i].round(5)
    out.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base", choices=list(SHORT_NAME))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--class_weights", default="sqrt_balanced", choices=["sqrt_balanced", "balanced", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test: stop after N steps")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    short = SHORT_NAME[args.model]
    save_dir = SAVE_ROOT / f"{short}_burnout"
    print(f"[1] {args.model} on {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(str(DATA_DIR / "tokenized" / args.model / "tokenizer"))
    pad_id = tokenizer.pad_token_id
    train, val, test = (load_split(args.model, s) for s in ["train", "val", "test"])
    meta = {s: pd.read_csv(DATA_DIR / f"{s}.csv", dtype={"student_id": str}) for s in ["val", "test"]}
    for s, d in [("val", val), ("test", test)]:          # parquet and csv rows must line up
        assert (meta[s]["student_id"].to_numpy() == d["student_id"].to_numpy()).all()
    max_len = int(train["input_ids"].map(len).max())
    print(f"[2] train {len(train):,} | val {len(val):,} | test {len(test):,} | max_len {max_len}")

    cw = json.loads((DATA_DIR / "class_weights.json").read_text())
    weight = None if args.class_weights == "none" else torch.tensor(cw[args.class_weights], dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID).to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": args.weight_decay},
              {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0}]
    optim = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total = args.max_steps or steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup * total), total)

    y_train = torch.tensor(train["label"].to_numpy(), device=device)
    lengths = train["input_ids"].map(len).to_numpy()
    history, best_f1, step = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        batches = length_bucketed_batches(lengths, args.batch_size, shuffle=True, seed=args.seed + epoch)
        for b, rows in enumerate(batches, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**collate(train, rows, pad_id, device)).logits
            loss = loss_fn(logits.float(), y_train[torch.as_tensor(rows, device=device)])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            run_loss += loss.item(); step += 1
            if b % 200 == 0:
                rate = b / (time.time() - t0)
                print(f"    epoch {epoch} step {b}/{len(batches)} loss {run_loss / b:.4f} "
                      f"({rate:.1f} it/s, ~{(len(batches) - b) / rate / 60:.1f} min left)", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        val_logits = predict(model, val, pad_id, device)
        val_scores = scores(val["label"].to_numpy(), val_logits.argmax(1))
        history.append({"epoch": epoch, "train_loss": round(run_loss / b, 4), **{f"val_{k}": v for k, v in val_scores.items()},
                        "minutes": round((time.time() - t0) / 60, 1)})
        print(f"[3] epoch {epoch}: train loss {run_loss / b:.4f} | val acc {val_scores['accuracy']} "
              f"macro-F1 {val_scores['f1_macro']} | {history[-1]['minutes']} min", flush=True)
        if val_scores["f1_macro"] > best_f1:
            best_f1 = val_scores["f1_macro"]
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"    saved best checkpoint -> {save_dir.relative_to(BASE_DIR)}")
        if args.max_steps and step >= args.max_steps:
            break

    # Test with the best checkpoint ---------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    val_logits = predict(model, val, pad_id, device)
    test_logits = predict(model, test, pad_id, device)
    report = full_report(meta["test"], test_logits.argmax(1))
    tag = f"weekly_v2_{short}" + ("_smoketest" if args.max_steps else "")
    (save_dir / "preprocessing.json").write_text(json.dumps({
        "max_len": max_len, "truncation": f"head+tail, first {HEAD_TOKENS} tokens + last {max_len - 2 - HEAD_TOKENS}",
        "cleaning": "preprocess_weekly.clean_text", "label2id": LABEL2ID}, indent=2))
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    save_predictions(PRED_DIR / f"{tag}_val_predictions.csv", val, meta["val"], val_logits)
    save_predictions(PRED_DIR / f"{tag}_test_predictions.csv", test, meta["test"], test_logits)
    (METRICS_DIR / f"{tag}_results.json").write_text(json.dumps({
        "model": args.model, "researcher": "IT22196392 - Induwara K.P.Y.",
        "dataset": "Combined_Data_all_weeks_v2 (student-level split, see data/processed/weekly_v2)",
        "hyperparameters": vars(args), "max_len": max_len, "history": history,
        "best_val_f1_macro": best_f1, "val": scores(val["label"].to_numpy(), val_logits.argmax(1)), "test": report}, indent=2))

    print(f"\n[4] TEST  acc {report['accuracy']} | macro-F1 {report['f1_macro']} "
          f"(95% CI {report['f1_macro_95ci_student_bootstrap']}) | weighted-F1 {report['f1_weighted']}")
    print("    per week macro-F1:", {w: v["f1_macro"] for w, v in report["per_week"].items()})
    print("    per class F1:", {k: v["f1"] for k, v in report["per_class"].items()})
    print("    at-risk (non-Normal) recall:", report["at_risk_binary"]["recall"])


if __name__ == "__main__":
    main()
