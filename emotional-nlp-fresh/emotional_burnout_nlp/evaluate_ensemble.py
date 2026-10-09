"""
Ensemble evaluation (Notebook 09, batched).

Same test split and same 50/50 soft voting as 09_ensemble_model.ipynb, but
batched on GPU instead of one row at a time, and the per-class numbers are
computed rather than hand-typed.

Also scores BERT-only and RoBERTa-only from the same forward passes, so the
three models are compared on identical rows at no extra cost.

    python evaluate_ensemble.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BERT_PATH = os.path.join(HERE, "models", "saved", "bert_burnout")
ROBERTA_PATH = os.path.join(HERE, "models", "saved", "roberta_burnout")
DATA_PATH = os.path.join(HERE, "data", "cleaned_data.csv")
OUT_PATH = os.path.join(HERE, "results", "metrics", "ensemble_results.json")

BATCH_SIZE = 64
MAX_LENGTH = 128
BERT_WEIGHT = 0.5
ROBERTA_WEIGHT = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------- test split
df = pd.read_csv(DATA_PATH).dropna(subset=["text_clean"])
df["text_clean"] = df["text_clean"].astype(str)

# Identical to notebooks 07/08/09: 70/30, then that 30 split in half.
# Test set is the final 15%. random_state=42 keeps it reproducible.
_, test_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label"])
_, test_df = train_test_split(
    test_df, test_size=0.50, random_state=42, stratify=test_df["label"])

print(f"Test samples: {len(test_df)}")


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path)
    mdl.to(device)
    mdl.eval()
    return tok, mdl


print("Loading BERT...")
bert_tokenizer, bert_model = load(BERT_PATH)
print("Loading RoBERTa...")
roberta_tokenizer, roberta_model = load(ROBERTA_PATH)

# Label order comes from the model config, not from df.unique(), so the
# mapping cannot silently drift if the CSV row order ever changes.
id2label = {int(k): v for k, v in bert_model.config.id2label.items()}
num_labels = len(id2label)
label_names = [id2label[i] for i in range(num_labels)]
assert {int(k): v for k, v in roberta_model.config.id2label.items()} == id2label, \
    "BERT and RoBERTa disagree on label order"
print(f"Labels: {label_names}")


@torch.no_grad()
def probs_for(tokenizer, model, texts):
    """Softmax probabilities for every text, in batches."""
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
        done = min(i + BATCH_SIZE, len(texts))
        if done % (BATCH_SIZE * 20) == 0 or done == len(texts):
            print(f"  {done}/{len(texts)}")
    return np.vstack(out)


texts = test_df["text_clean"].tolist()
y_true = test_df["label"].tolist()

print("Running BERT...")
bert_probs = probs_for(bert_tokenizer, bert_model, texts)
print("Running RoBERTa...")
roberta_probs = probs_for(roberta_tokenizer, roberta_model, texts)

# 50/50 soft voting - average the two probability distributions.
ensemble_probs = BERT_WEIGHT * bert_probs + ROBERTA_WEIGHT * roberta_probs


def score(probs, title):
    y_pred = [id2label[i] for i in probs.argmax(axis=1)]
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n=== {title} ===")
    print(f"Accuracy    : {acc:.4f}")
    print(f"F1 weighted : {f1_w:.4f}")
    print(f"F1 macro    : {f1_m:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return y_pred, acc, f1_w, f1_m


score(bert_probs, "BERT only")
score(roberta_probs, "RoBERTa only")
y_pred, acc, f1_w, f1_m = score(ensemble_probs, "ENSEMBLE (soft voting)")

# --------------------------------------------------------- real per-class
precision, recall, f1_pc, support = precision_recall_fscore_support(
    y_true, y_pred, labels=label_names, zero_division=0)

results = {
    "model": "BERT + RoBERTa Ensemble (Soft Voting)",
    "accuracy": round(float(acc), 4),
    "f1_score": round(float(f1_w), 4),
    "f1_macro": round(float(f1_m), 4),
    "bert_weight": BERT_WEIGHT,
    "roberta_weight": ROBERTA_WEIGHT,
    "test_samples": int(len(test_df)),
    "per_class": {
        label_names[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1_pc[i]), 4),
            "support": int(support[i]),
        }
        for i in range(num_labels)
    },
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT_PATH}")
