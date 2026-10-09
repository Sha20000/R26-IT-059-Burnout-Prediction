"""
Stage 5 — Weekly emotional stress scores for the meta layer.
IT22196392 — Induwara K.P.Y. | R26-IT-059

Scores every student-week of the weekly dataset v2 with the BERT + RoBERTa ensemble
(same weights as serve_model.py) and writes:

  results/weekly_v2_emotional_stress_weekly.csv   one row per student-week
  results/weekly_v2_emotional_stress_scores.csv   one row per student, in the same format as
                                                  results/emotional_stress_scores.csv (what
                                                  meta-integration reads) + weekly columns
  results/metrics/weekly_v2_stress_export_summary.json

emotional_stress_score uses the meta layer's 0-4 scale (preprocess_weekly.STRESS_LEVEL).
Per student it is the mean of the weekly scores; peak and trend are given too.

Note: students in the NLP *train* split were seen during fine-tuning, so their predictions
are more accurate than real-world ones. The `nlp_split` column marks this; only `test`
students are fully out-of-sample.

Needs the GPU env:  C:/Users/ASUS/.venvs/burnout-nlp-gpu/Scripts/python.exe export_weekly_stress_scores.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from preprocess_weekly import ID2LABEL, OUT_DIR as DATA_DIR, STRESS_LEVEL
from train_baseline_weekly import RISK
from train_transformer_weekly import load_split, predict

BASE_DIR = Path(__file__).parent
RESULTS = BASE_DIR / "results"
MODELS = {"bert": "bert-base-uncased", "roberta": "roberta-base"}
LABELS = [ID2LABEL[i] for i in range(len(ID2LABEL))]
LEVEL = np.array([STRESS_LEVEL[l] for l in LABELS], dtype=float)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = pd.read_csv(DATA_DIR / "weekly_clean.csv", dtype={"student_id": str})
    ens = json.loads((RESULTS / "metrics" / "weekly_v2_ensemble_results.json").read_text())["ensemble"]
    weights = {"bert": ens["bert_weight"], "roberta": ens["roberta_weight"]}
    print(f"[1] {len(rows):,} student-weeks, {rows.student_id.nunique():,} students | weights {weights} | {device}")

    probs = np.zeros((len(rows), len(LABELS)))
    for short, hf_name in MODELS.items():
        tok = pd.concat([load_split(hf_name, s) for s in ["train", "val", "test"]], ignore_index=True)
        tok = rows[["student_id", "week"]].merge(tok, on=["student_id", "week"], how="left", validate="one_to_one")
        assert tok["input_ids"].notna().all()
        model = AutoModelForSequenceClassification.from_pretrained(
            str(BASE_DIR / "models" / "saved" / "weekly_v2" / f"{short}_burnout")).to(device)
        logits = predict(model, tok, model.config.pad_token_id, device, batch_size=64)   # 128 ran out of memory on 8 GB
        probs += weights[short] * torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        print(f"[2] {short} scored")
        del model
        torch.cuda.empty_cache()

    pred = probs.argmax(1)
    weekly = rows[["student_id", "week", "split"]].rename(columns={"split": "nlp_split"})
    weekly["dataset_label"] = rows["status"]
    weekly["predicted_label"] = [LABELS[i] for i in pred]
    weekly["confidence"] = probs.max(1).round(4)
    weekly["risk_level"] = weekly["predicted_label"].map(RISK)
    weekly["emotional_stress_score"] = weekly["predicted_label"].map(STRESS_LEVEL)
    weekly["emotional_stress_expected"] = (probs @ LEVEL).round(4)          # probability-weighted, 0-4
    for i, lab in enumerate(LABELS):
        weekly[f"p_{lab}"] = probs[:, i].round(5)
    weekly["text_sample"] = rows["text_clean"].str.slice(0, 100)

    # Consistency: the test rows must match ensemble_weekly.py's predictions.
    ref = pd.read_csv(RESULTS / "predictions" / "weekly_v2_ensemble_test_predictions.csv", dtype={"student_id": str})
    chk = ref.merge(weekly, on=["student_id", "week"])
    agree = float(chk["pred_label"].eq(chk["predicted_label"]).mean())
    print(f"[3] agreement with ensemble_weekly.py on test rows: {agree:.4f}")

    weekly = weekly.sort_values(["student_id", "week"]).reset_index(drop=True)
    weekly.to_csv(RESULTS / "weekly_v2_emotional_stress_weekly.csv", index=False)

    # Per student, meta-layer format ------------------------------------------------------
    g = weekly.groupby("student_id", sort=True)
    latest = g.tail(1).set_index("student_id")
    wide = weekly.pivot(index="student_id", columns="week", values="emotional_stress_score")
    wide.columns = [f"week{w}_emotional_stress" for w in wide.columns]
    first_last = g["emotional_stress_score"].agg(["first", "last"])
    per = pd.DataFrame({
        # Unpadded number (STU00012 -> STU12): meta_intregation.py matches students on the
        # numeric suffix compared as a plain integer string.
        "student_id": "STU" + latest.index.str.extract(r"(\d+)$", expand=False).astype(int).astype(str),
        "text_sample": latest["text_sample"].to_numpy(),
        "predicted_label": latest["predicted_label"].to_numpy(),
        "confidence": latest["confidence"].to_numpy(),
        "emotional_stress_score": g["emotional_stress_score"].mean().round(4).to_numpy(),
        "risk_level": latest["risk_level"].to_numpy(),
    })
    per = pd.concat([per.reset_index(drop=True), wide.reset_index(drop=True)], axis=1)
    per["emotional_stress_peak"] = g["emotional_stress_score"].max().to_numpy()
    per["emotional_stress_trend"] = (first_last["last"] - first_last["first"]).to_numpy()   # last week - first week
    per["emotional_stress_expected_mean"] = g["emotional_stress_expected"].mean().round(4).to_numpy()
    per["n_weeks"] = g.size().to_numpy()
    per["nlp_student_id"] = latest.index.to_numpy()
    per["nlp_split"] = latest["nlp_split"].to_numpy()
    per.to_csv(RESULTS / "weekly_v2_emotional_stress_scores.csv", index=False)

    # Summary ------------------------------------------------------------------------------
    acc = {s: round(float((d["predicted_label"] == d["dataset_label"]).mean()), 4) for s, d in weekly.groupby("nlp_split")}
    by_week = weekly.groupby("week")["emotional_stress_score"].agg(["mean", "std"]).round(3)
    summary = {
        "student_weeks": len(weekly), "students": len(per), "ensemble_weights": weights,
        "agreement_with_ensemble_test_predictions": round(agree, 4),
        "accuracy_by_nlp_split": acc,
        "note": "train-split students were seen in fine-tuning; only test-split scores are out-of-sample",
        "stress_score_by_week": {int(w): {"mean": float(r["mean"]), "std": float(r["std"])} for w, r in by_week.iterrows()},
        "per_student_score": {"definition": "mean of weekly 0-4 scores", "mean": round(float(per.emotional_stress_score.mean()), 3),
                              "students_with_peak_4": int((per.emotional_stress_peak == 4).sum())},
        "meta_layer_id_match": "student_id is unpadded (STU12) so meta_intregation.py's numeric-suffix merge works",
    }
    (RESULTS / "metrics" / "weekly_v2_stress_export_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[4] accuracy by split: {acc}")
    print(f"    mean weekly score: { {int(w): float(r['mean']) for w, r in by_week.iterrows()} }")
    print(f"[5] Saved {len(weekly):,} weekly rows and {len(per):,} student rows to results/")


if __name__ == "__main__":
    main()
