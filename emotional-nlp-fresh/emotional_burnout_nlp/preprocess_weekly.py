"""
Stage 1 — Data preprocessing for the weekly emotional-NLP dataset.
IT22196392 — Induwara K.P.Y. | R26-IT-059

Input : data/Combined_Data_all_weeks_v2.csv   (student_id, week, statement, status)
Output: data/processed/weekly_v2/
          weekly_clean.csv              all kept rows + clean text + label_id + split
          dropped_rows.csv              rows removed and why
          train.csv / val.csv / test.csv
          tokenized/<model>/{train,val,test}.parquet   input_ids, attention_mask, label
          label_map.json, class_weights.json, preprocessing_report.json
        results/figures/weekly_label_distribution.png
        results/figures/token_length_distribution.png

Run:  .venv/Scripts/python.exe preprocess_weekly.py

The functions clean_text() and encode_head_tail() are meant to be imported by the
training and serving code, so a text is processed the same way everywhere.
"""
import html
import json
import re
from pathlib import Path

import ftfy
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
RAW_CSV = BASE_DIR / "data" / "Combined_Data_all_weeks_v2.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "weekly_v2"
FIG_DIR = BASE_DIR / "results" / "figures"

SEED = 42
TOKENIZERS = ["bert-base-uncased", "roberta-base"]   # the two ensemble members
MAX_LEN_CHOICES = [128, 192, 256]                    # agreed range 128–256
HEAD_TOKENS = 128                                    # head+tail truncation (Sun et al., 2019)

# Same ids as the saved bert_burnout / roberta_burnout models, so serve_model.py stays compatible.
LABEL2ID = {"Anxiety": 0, "Normal": 1, "Depression": 2, "Suicidal": 3,
            "Stress": 4, "Bipolar": 5, "Personality disorder": 6}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# 0-4 emotional stress score the meta layer reads (it divides by 4). Same mapping as the
# original results/emotional_stress_scores.csv, so the Meta-FNN input keeps its meaning.
STRESS_LEVEL = {"Normal": 0, "Stress": 1, "Anxiety": 2, "Depression": 3,
                "Bipolar": 3, "Personality disorder": 3, "Suicidal": 4}


# ----------------------------------------------------------------------------- cleaning
_URL = re.compile(r"(https?://\S+|www\.\S+|\bt\.co/\S+|\bhttps?\s+t\s+co\s+\S+)", re.I)
_MENTION = re.compile(r"(?<!\w)@\w+")
_HASHTAG = re.compile(r"(?<!\w)#(\w+)")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_X200B = re.compile(r"(&#x200b;|\bamp\s*x\s*200b\b|\bamp\s*x\s*00b\b|\bx200b\b)", re.I)
_MD_ESCAPE = re.compile(r"\\(?=[^\w\s])")             # "\---" -> "---", "\*" -> "*"
_MARKDOWN = re.compile(r"(\*\*|__|~~|`{1,3}|^\s*>+\s?|(?<!\S)-{3,}(?!\S))", re.M)
_ELONGATED = re.compile(r"([^\d\s])\1{3,}")           # "wtffffff" -> "wtfff", "!!!!!!" -> "!!!" (numbers untouched)
_MOJIBAKE_LEFT = re.compile(r"â€[^\sA-Za-z0-9]?|ï¸|Â(?=\s|$)")  # fragments ftfy cannot repair (bytes already lost)
_EMOJI_MOJIBAKE = re.compile(r"ðŸ[^\sA-Za-z0-9ð]{0,2}")         # one broken emoji, e.g. "ðŸ¤£"


def _fix_emoji(match) -> str:
    """Repair one broken emoji on its own (ftfy gives up when a text mixes good and
    damaged ones); drop it if a byte is lost, since ftfy would only guess which emoji it was."""
    piece = match.group(0)
    if len(piece) < 4:
        return " "
    fixed = ftfy.fix_text(piece)
    return fixed if "ð" not in fixed and "Ÿ" not in fixed else " "
# Part of the source was pre-stripped of apostrophes ("don t", "i m"); put them back.
_CONTRACTIONS = [
    (re.compile(r"\b(do|does|did|is|are|was|were|has|have|had|could|would|should|must|need|wo|ca|ai)n t\b", re.I), r"\1n't"),
    (re.compile(r"\b(i) m\b", re.I), r"\1'm"),
    (re.compile(r"\b(i|you|we|they) (ve|re|ll|d)\b", re.I), r"\1'\2"),
    (re.compile(r"\b(it|that|there|what|he|she|let) s\b", re.I), r"\1's"),
]
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Light, transformer-friendly cleaning.

    Keeps case, punctuation, apostrophes, digits and emoji, because BERT/RoBERTa use them
    ("can't" vs "cant", "!!!", "😭"). Only removes noise that carries no emotion.
    """
    text = ftfy.fix_text(str(text))                    # broken characters: "donâ€™t" -> "don’t"
    text = _EMOJI_MOJIBAKE.sub(_fix_emoji, text)
    text = _MOJIBAKE_LEFT.sub(" ", text)
    text = html.unescape(text)                        # "&amp;" -> "&", "&#x200B;" -> zero-width
    text = _X200B.sub(" ", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(r"\1", text)                   # keep the word, drop the '#'
    text = _MD_ESCAPE.sub("", text)
    text = _MARKDOWN.sub(" ", text)
    text = _ELONGATED.sub(r"\1\1\1", text)
    for pattern, repl in _CONTRACTIONS:
        text = pattern.sub(repl, text)
    return _SPACES.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    """Case/punctuation-insensitive key used to find duplicate texts."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ----------------------------------------------------------------------------- tokenizing
def encode_head_tail(tokenizer, texts, max_len, head=HEAD_TOKENS):
    """Tokenize without padding. Long texts keep the first `head` tokens and the last
    (max_len - 2 - head) tokens instead of cutting the end off — the end of a post often
    holds the key sentence ("...I want to die", "please help")."""
    budget = max_len - 2                               # room for [CLS]/[SEP] or <s>/</s>
    head = min(head, budget)
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    ids_all = tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    out_ids, raw_len = [], []
    for ids in ids_all:
        raw_len.append(len(ids))
        if len(ids) > budget:
            ids = ids[:head] + ids[len(ids) - (budget - head):]
        out_ids.append([cls] + ids + [sep])
    masks = [[1] * len(x) for x in out_ids]
    return out_ids, masks, np.array(raw_len)


# ----------------------------------------------------------------------------- splitting
def link_students_by_shared_text(df):
    """Union-find: students that share an identical text become one group, so a text
    can never appear in both train and test."""
    parent = {s: s for s in df["student_id"].unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, students in df.groupby("dedup_key")["student_id"]:
        students = students.unique()
        root = find(students[0])
        for s in students[1:]:
            r = find(s)
            if r != root:
                parent[r] = root
    return df["student_id"].map(lambda s: find(s))


def group_split(df, seed=SEED):
    """70/15/15 split by group, stratified by label (StratifiedGroupKFold, 20 folds:
    14 train, 3 val, 3 test)."""
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label_id"], groups=df["group_id"])):
        fold[idx] = k
    return np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))


# ----------------------------------------------------------------------------- main
def main():
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()                   # "sequence longer than 512" is expected: we truncate ourselves

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report = {"input_file": RAW_CSV.name, "seed": SEED}

    # 1. Load ----------------------------------------------------------------------------
    df = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    df["student_id"] = df["student_id"].str.strip()
    df["status"] = df["status"].str.strip()
    report["raw_rows"] = len(df)
    report["raw_students"] = int(df["student_id"].nunique())
    print(f"[1] Loaded {len(df):,} rows, {df['student_id'].nunique():,} students, weeks {sorted(int(w) for w in df['week'].unique())}")

    unknown = set(df["status"].dropna()) - set(LABEL2ID)
    assert not unknown, f"Unknown labels: {unknown}"

    dropped = []

    def drop(mask, reason):
        nonlocal df
        if mask.any():
            dropped.append(df[mask].assign(drop_reason=reason))
            df = df[~mask].copy()
        report.setdefault("dropped", {})[reason] = int(mask.sum())

    drop(df["statement"].isna() | df["status"].isna(), "missing text or label")
    drop(df["statement"].astype(str).str.match(r"^\s*#(NAME\?|VALUE!|REF!|N/A|DIV/0!|NUM!|NULL!)\s*$"),
         "Excel error value instead of text (e.g. #NAME?)")
    drop(df.duplicated(["student_id", "week"], keep="first"), "duplicate student-week")

    # 2. Clean ---------------------------------------------------------------------------
    df["text_clean"] = df["statement"].map(clean_text)
    df["dedup_key"] = df["text_clean"].map(dedup_key)
    changed = (df["text_clean"] != df["statement"].astype(str).str.strip()).sum()
    report["rows_changed_by_cleaning"] = int(changed)
    print(f"[2] Cleaned text — {changed:,} rows changed (original kept in 'statement')")

    drop(~df["text_clean"].str.contains(r"[A-Za-z]{2,}", regex=True), "no readable words after cleaning")

    # Same text, different labels = label noise; it cannot be learned either way.
    n_labels = df.groupby("dedup_key")["status"].transform("nunique")
    drop(n_labels > 1, "same text with conflicting labels")

    # 3. Labels → numbers ----------------------------------------------------------------
    df["label_id"] = df["status"].map(LABEL2ID).astype(int)
    df["word_count"] = df["text_clean"].str.split().str.len()
    dup_size = df.groupby("dedup_key")["dedup_key"].transform("size")
    df["is_duplicate_text"] = dup_size > 1
    report["duplicate_text_rows_kept"] = int(df["is_duplicate_text"].sum())
    print(f"[3] Labels mapped: {LABEL2ID}")

    # 4. Split 70/15/15 by student group ------------------------------------------------
    df["group_id"] = link_students_by_shared_text(df)
    gsize = df.groupby("group_id")["student_id"].nunique()
    report["groups"] = {"count": int(len(gsize)), "multi_student_groups": int((gsize > 1).sum()),
                        "largest_group_students": int(gsize.max())}
    df["split"] = group_split(df)
    df = df.sort_values(["split", "student_id", "week"]).reset_index(drop=True)

    # Leakage checks — these must all be zero.
    def overlap(col, a, b):
        return len(set(df.loc[df.split == a, col]) & set(df.loc[df.split == b, col]))

    leak = {f"{col}:{a}-{b}": overlap(col, a, b)
            for col in ["student_id", "dedup_key"]
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    assert all(v == 0 for v in leak.values()), f"Leakage found: {leak}"
    report["leakage_checks"] = leak

    split_stats = {}
    for name, part in df.groupby("split", sort=False):
        split_stats[name] = {
            "rows": len(part), "share": round(len(part) / len(df), 4),
            "students": int(part["student_id"].nunique()),
            "label_counts": part["status"].value_counts().to_dict(),
            "label_share": part["status"].value_counts(normalize=True).round(4).to_dict(),
            "week_counts": {int(k): int(v) for k, v in part["week"].value_counts().sort_index().items()},
        }
        assert part["label_id"].nunique() == len(LABEL2ID), f"{name} is missing a class"
        assert part["week"].nunique() == df["week"].nunique(), f"{name} is missing a week"
    report["splits"] = split_stats
    for s in ["train", "val", "test"]:
        st = split_stats[s]
        print(f"[4] {s:5s}: {st['rows']:6,} rows ({st['share']:.1%}), {st['students']:,} students")

    # 5. Imbalance — class weights from TRAIN only ---------------------------------------
    y_train = df.loc[df.split == "train", "label_id"].to_numpy()
    classes = np.arange(len(LABEL2ID))
    balanced = compute_class_weight("balanced", classes=classes, y=y_train)
    sqrt_bal = np.sqrt(balanced)
    sqrt_bal = sqrt_bal / (sqrt_bal * np.bincount(y_train, minlength=len(classes))).sum() * len(y_train)
    class_weights = {
        "note": "Index = label_id. Use in torch.nn.CrossEntropyLoss(weight=...). Computed on train split only.",
        "label_order": [ID2LABEL[i] for i in classes],
        "train_counts": np.bincount(y_train, minlength=len(classes)).tolist(),
        "balanced": np.round(balanced, 4).tolist(),
        "sqrt_balanced": np.round(sqrt_bal, 4).tolist(),
        "recommended": "sqrt_balanced",
        "why": "Full 'balanced' weights push rare classes ~7x; the square-root version is gentler "
               "and usually more stable for transformer fine-tuning. Select the model on macro-F1.",
    }
    print("[5] Class weights (sqrt_balanced):",
          {ID2LABEL[i]: round(float(w), 2) for i, w in enumerate(sqrt_bal)})

    # 6. Tokenize + pick max length ------------------------------------------------------
    token_stats, chosen, train_lens = {}, {}, {}
    for model_name in TOKENIZERS:
        tok = AutoTokenizer.from_pretrained(model_name)
        tr = df[df.split == "train"]
        _, _, raw = encode_head_tail(tok, tr["text_clean"], max_len=512)
        lens = raw + 2
        cover = {L: round(float((lens <= L).mean()), 4) for L in MAX_LEN_CHOICES + [512]}
        per_class = {ID2LABEL[i]: int(np.median(lens[tr["label_id"].to_numpy() == i])) for i in classes}
        # Smallest length in range that keeps >= 80% of train texts whole, else the top of the range.
        max_len = next((L for L in MAX_LEN_CHOICES if cover[L] >= 0.80), MAX_LEN_CHOICES[-1])
        chosen[model_name] = max_len
        token_stats[model_name] = {
            "percentiles": {f"p{p}": int(np.percentile(lens, p)) for p in [50, 75, 90, 95, 99]},
            "coverage_without_truncation": {str(k): v for k, v in cover.items()},
            "median_tokens_per_class": per_class,
            "chosen_max_len": max_len,
            "truncation": f"head+tail (first {HEAD_TOKENS} + last {max_len - 2 - HEAD_TOKENS} tokens)",
        }
        print(f"[6] {model_name}: median {int(np.median(lens))} tokens, "
              f"fits in 128={cover[128]:.0%} 256={cover[256]:.0%} -> max_len {max_len}")

        out = OUT_DIR / "tokenized" / model_name
        out.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            part = df[df.split == split]
            ids, masks, raw_len = encode_head_tail(tok, part["text_clean"], max_len)
            assert max(len(x) for x in ids) <= max_len
            pd.DataFrame({
                "student_id": part["student_id"].to_numpy(), "week": part["week"].to_numpy(),
                "label": part["label_id"].to_numpy(), "input_ids": ids, "attention_mask": masks,
                "n_tokens_before_truncation": raw_len + 2,
            }).to_parquet(out / f"{split}.parquet", index=False)
        tok.save_pretrained(out / "tokenizer")
        token_stats[model_name]["train_share_truncated"] = round(float((lens > max_len).mean()), 4)
        train_lens[model_name] = lens
    report["tokenization"] = token_stats

    # 7. Save ----------------------------------------------------------------------------
    keep = ["student_id", "week", "statement", "text_clean", "status", "label_id",
            "split", "group_id", "word_count", "is_duplicate_text"]
    df[keep].to_csv(OUT_DIR / "weekly_clean.csv", index=False, encoding="utf-8")
    for s in ["train", "val", "test"]:
        df.loc[df.split == s, ["student_id", "week", "text_clean", "status", "label_id"]] \
          .to_csv(OUT_DIR / f"{s}.csv", index=False, encoding="utf-8")
    if dropped:
        pd.concat(dropped)[["student_id", "week", "statement", "status", "drop_reason"]] \
          .to_csv(OUT_DIR / "dropped_rows.csv", index=False, encoding="utf-8")

    (OUT_DIR / "label_map.json").write_text(json.dumps(
        {"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, indent=2))
    (OUT_DIR / "class_weights.json").write_text(json.dumps(class_weights, indent=2))
    report["final_rows"] = len(df)
    report["final_students"] = int(df["student_id"].nunique())
    report["chosen_max_len"] = chosen
    report["median_words_per_class"] = df.groupby("status")["word_count"].median().astype(int).to_dict()
    report["label_by_week_all"] = {str(k): v for k, v in
                                   pd.crosstab(df["status"], df["week"]).to_dict().items()}
    (OUT_DIR / "preprocessing_report.json").write_text(json.dumps(report, indent=2, default=int))

    # 8. Figures -------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    share = pd.crosstab(df["week"], df["status"], normalize="index")[list(LABEL2ID)]
    share.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab10", width=0.75)
    axes[0].set_title("Label share by semester week")
    axes[0].set_xlabel("Week"); axes[0].set_ylabel("Share of texts")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    split_share = pd.crosstab(df["status"], df["split"], normalize="columns")[["train", "val", "test"]]
    split_share.loc[list(LABEL2ID)].plot(kind="bar", ax=axes[1], width=0.8)
    axes[1].set_title("Label share per split (stratified group split)")
    axes[1].set_ylabel("Share"); axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Weekly dataset v2 — IT22196392", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weekly_label_distribution.png", dpi=150)
    plt.close(fig)

    tr = df[df.split == "train"]
    lens = train_lens[TOKENIZERS[1]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for lab in LABEL2ID:
        ax.hist(np.clip(lens[tr["status"].to_numpy() == lab], 0, 600), bins=60, alpha=0.45, label=lab)
    for L in MAX_LEN_CHOICES:
        ax.axvline(L, ls="--", c="k" if L == chosen[TOKENIZERS[1]] else "grey", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("RoBERTa tokens per text (clipped at 600)"); ax.set_ylabel("Texts (log)")
    ax.set_title("Token length by class — train split (dashed: 128 / 192 / 256)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "token_length_distribution.png", dpi=150)
    plt.close(fig)

    print(f"[7] Saved to {OUT_DIR.relative_to(BASE_DIR)}  ({len(df):,} rows, "
          f"{sum(report['dropped'].values())} dropped)")
    print(f"[8] Figures saved to {FIG_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
