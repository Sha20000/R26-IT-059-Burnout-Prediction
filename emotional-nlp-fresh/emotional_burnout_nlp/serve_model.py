#!/usr/bin/env python3
"""
Emotional Burnout NLP — warm inference service
IT22196392 — Induwara K.P.Y. | R26-IT-059

Loads the fine-tuned BERT *and* RoBERTa models ONCE at startup and keeps them in
memory, instead of re-loading the weights on every request the way predict.py
does when spawned per-call.

Models: the weekly-dataset-v2 retrain (models/saved/weekly_v2, trained by
train_transformer_weekly.py on a student-level split). Text goes through the same
clean_text() and 256-token head+tail encoding used in training (preprocess_weekly.py).

Prediction is a soft-voting ensemble: both models score the text, their softmax
probabilities are mixed with the weights chosen on validation by ensemble_weekly.py
(RoBERTa 0.55 / BERT 0.45), and the highest mixed probability wins.

Attention (the XAI token highlighting) is taken from BERT only. BERT and
RoBERTa use different tokenizers - WordPiece vs BPE - so their tokens do not
line up one-to-one and their attention cannot be averaged meaningfully.

Serves the integrated dashboard (Emotional Analysis tab) on port 5004,
alongside the other members' APIs:
    5001  academic / burnout        (my-academic-gru)
    5002  meta-integration
    5003  behavioural anomaly VAE
    5004  emotional NLP             <- this service
"""

import json
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from preprocess_weekly import HEAD_TOKENS, STRESS_LEVEL, clean_text, encode_head_tail

app = Flask(__name__)
CORS(app)  # dashboard is served from a different origin (:8080)

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models/saved/weekly_v2"
BERT_PATH = MODEL_DIR / "bert_burnout"
ROBERTA_PATH = MODEL_DIR / "roberta_burnout"
METRICS_PATH = HERE / "results/metrics/weekly_v2_ensemble_results.json"
PORT = 5004

# Soft-voting weights. Replaced at startup by the ones ensemble_weekly.py chose on validation.
BERT_WEIGHT = 0.45
ROBERTA_WEIGHT = 0.55
MAX_LEN = 256             # replaced at startup from the model's preprocessing.json

# Same mapping as predict.py
RISK_MAP = {
    "Normal": "Low",
    "Stress": "Medium",
    "Anxiety": "Medium",
    "Depression": "High",
    "Bipolar": "High",
    "Personality disorder": "High",
    "Suicidal": "Critical",
}

# Weight each class contributes to the 0-100 distress score (from predict.py)
STRESS_WEIGHTS = {
    "Suicidal": 1.0,
    "Depression": 0.8,
    "Bipolar": 0.75,
    "Personality disorder": 0.7,
    "Anxiety": 0.6,
    "Stress": 0.5,
}

TOKENIZER = None          # BERT tokenizer
MODEL = None              # BERT model (also the source of attention)
ROBERTA_TOKENIZER = None
ROBERTA_MODEL = None
LABELS = []               # index -> label name, from the model config


def load_model():
    """Load both tokenizers + models once into module globals."""
    global TOKENIZER, MODEL, ROBERTA_TOKENIZER, ROBERTA_MODEL, LABELS
    global BERT_WEIGHT, ROBERTA_WEIGHT, MAX_LEN

    for name, path in (("BERT", BERT_PATH), ("RoBERTa", ROBERTA_PATH)):
        if not path.exists():
            print(f"[X] {name} model not found at {path}")
            return False

    try:
        with open(METRICS_PATH) as f:
            ens = json.load(f)["ensemble"]
        BERT_WEIGHT, ROBERTA_WEIGHT = ens["bert_weight"], ens["roberta_weight"]
    except (OSError, KeyError, ValueError) as e:
        print(f"[!] Ensemble weights not found ({e}) - using defaults")
    with open(BERT_PATH / "preprocessing.json") as f:
        MAX_LEN = json.load(f)["max_len"]

    print("[..] Loading BERT + RoBERTa (one-time, ~30s)")
    try:
        # output_attentions only on BERT - it is what drives the XAI view.
        TOKENIZER = AutoTokenizer.from_pretrained(str(BERT_PATH))
        MODEL = AutoModelForSequenceClassification.from_pretrained(
            str(BERT_PATH), output_attentions=True)
        MODEL.eval()

        ROBERTA_TOKENIZER = AutoTokenizer.from_pretrained(str(ROBERTA_PATH))
        ROBERTA_MODEL = AutoModelForSequenceClassification.from_pretrained(
            str(ROBERTA_PATH))
        ROBERTA_MODEL.eval()
    except Exception as e:
        print(f"[X] Failed to load models: {e}")
        return False

    bert_labels = {int(k): v for k, v in MODEL.config.id2label.items()}
    roberta_labels = {int(k): v for k, v in ROBERTA_MODEL.config.id2label.items()}
    if bert_labels != roberta_labels:
        # Averaging probabilities across models whose class order differs would
        # silently produce nonsense, so refuse to start instead.
        print("[X] BERT and RoBERTa disagree on label order - cannot ensemble.")
        print(f"    BERT   : {bert_labels}")
        print(f"    RoBERTa: {roberta_labels}")
        return False

    LABELS = [bert_labels[i] for i in range(len(bert_labels))]
    # Long posts are tokenized in full before head+tail truncation, which makes the
    # tokenizer warn "sequence longer than 512" - expected, so keep the log clean.
    hf_logging.set_verbosity_error()
    print(f"[OK] Ensemble loaded — {len(LABELS)} classes: {', '.join(LABELS)}")
    print(f"     Soft voting: BERT {BERT_WEIGHT} / RoBERTa {ROBERTA_WEIGHT} | max_len {MAX_LEN} (head+tail)")
    return True


def encode(tokenizer, text):
    """Same encoding as training: head+tail truncation to MAX_LEN, no padding."""
    ids, mask, raw_len = encode_head_tail(tokenizer, [text], MAX_LEN)
    truncated = bool(raw_len[0] + 2 > MAX_LEN)
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}, truncated


def analyze_text(text):
    """Run the ensemble. Returns the same dict shape predict.py prints."""
    text = clean_text(text)
    inputs, truncated = encode(TOKENIZER, text)
    roberta_inputs, _ = encode(ROBERTA_TOKENIZER, text)

    with torch.no_grad():
        outputs = MODEL(**inputs)                       # attentions come from here
        roberta_outputs = ROBERTA_MODEL(**roberta_inputs)

    bert_probs = torch.softmax(outputs.logits, dim=1)[0]
    roberta_probs = torch.softmax(roberta_outputs.logits, dim=1)[0]

    # Soft voting - average the two distributions, then take the argmax.
    probs = BERT_WEIGHT * bert_probs + ROBERTA_WEIGHT * roberta_probs

    predicted_id = int(probs.argmax().item())
    predicted_label = LABELS[predicted_id]
    confidence = float(probs.max().item())

    probabilities = {
        LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)
    }

    # What each model said on its own - lets the demo show the ensemble working,
    # and makes disagreement between the two visible rather than hidden.
    model_breakdown = {
        "bert": {
            "prediction": LABELS[int(bert_probs.argmax().item())],
            "confidence": round(float(bert_probs.max().item()), 4),
        },
        "roberta": {
            "prediction": LABELS[int(roberta_probs.argmax().item())],
            "confidence": round(float(roberta_probs.max().item()), 4),
        },
    }
    model_breakdown["agree"] = (
        model_breakdown["bert"]["prediction"]
        == model_breakdown["roberta"]["prediction"]
    )

    stress_score = round(
        sum(probabilities.get(cls, 0) * w for cls, w in STRESS_WEIGHTS.items()) * 100,
        1,
    )

    # Attention over the [CLS] row, averaged across layers and heads
    all_attentions = torch.stack(outputs.attentions)
    avg_attention = all_attentions.mean(dim=0).mean(dim=1).squeeze()
    cls_attention = avg_attention[0].numpy()
    tokens = TOKENIZER.convert_ids_to_tokens(inputs["input_ids"][0])

    clean_tokens = tokens[1:-1]          # strip [CLS] / [SEP]
    clean_weights = cls_attention[1:-1]
    total = clean_weights.sum()
    if total > 0:
        clean_weights = clean_weights / total

    attention = [
        {"token": t[2:] if t.startswith("##") else t, "weight": round(float(w), 6)}
        for t, w in zip(clean_tokens, clean_weights)
    ]
    if truncated:
        # Long text: the middle was skipped (head+tail). Mark the gap so the
        # highlighted tokens are not read as one continuous sentence.
        attention.insert(HEAD_TOKENS, {"token": "…", "weight": 0.0})

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
        "risk_level": RISK_MAP.get(predicted_label, "Unknown"),
        "stress_score": stress_score,
        "emotional_stress_score": STRESS_LEVEL.get(predicted_label, 0),   # 0-4, meta-layer scale
        "probabilities": probabilities,
        "attention": attention,
        "text_truncated": truncated,
        "model": "BERT + RoBERTa ensemble (soft voting, weekly v2)",
        "models": model_breakdown,
    }


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    loaded = MODEL is not None and ROBERTA_MODEL is not None
    return jsonify({
        "status": "online" if loaded else "model_not_loaded",
        "service": "emotional-nlp",
        "model": "BERT fine-tuned + RoBERTa ensemble (soft voting)",
        "model_version": "weekly_v2",
        "ensemble": {
            "bert_loaded": MODEL is not None,
            "roberta_loaded": ROBERTA_MODEL is not None,
            "bert_weight": BERT_WEIGHT,
            "roberta_weight": ROBERTA_WEIGHT,
            "max_len": MAX_LEN,
        },
        "classes": LABELS,
        "api_version": "2.1",
    })


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return ("", 200)

    if MODEL is None:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    # Same validation as predict.py so behaviour matches the original frontend
    if not text or len(text.split()) < 3:
        return jsonify({"error": "Please enter at least a few words"}), 400

    try:
        return jsonify(analyze_text(text))
    except Exception as e:
        return jsonify({"error": "Model inference failed.", "detail": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Metrics shown in the dashboard's model performance panel.

    Read from results/metrics/weekly_v2_ensemble_results.json, which
    ensemble_weekly.py writes, so the dashboard always reports the measured
    numbers rather than figures typed in by hand.
    """
    info = {
        "model": "BERT Fine-tuned + RoBERTa Ensemble (soft voting)",
        "model_version": "weekly_v2",
        "num_classes": len(LABELS),
        "classes": LABELS,
        "author": "IT22196392 — Induwara K.P.Y.",
        "project": "R26-IT-059",
    }

    try:
        with open(METRICS_PATH) as f:
            m = json.load(f)
        t, ens = m["test"], m["ensemble"]
        info.update({
            "accuracy": f"{t['accuracy'] * 100:.2f}%",
            "f1_score": f"{t['f1_weighted'] * 100:.2f}%",
            "f1_macro": f"{t['f1_macro'] * 100:.2f}%",
            "f1_macro_95ci": t.get("f1_macro_95ci_student_bootstrap"),
            "test_samples": sum(v["support"] for v in t["per_class"].values()),
            "bert_weight": ens["bert_weight"],
            "roberta_weight": ens["roberta_weight"],
            "per_class_f1": {
                cls: f"{vals['f1'] * 100:.0f}%"
                for cls, vals in t["per_class"].items()
            },
            "per_class": t["per_class"],
            "per_week_f1_macro": {w: v["f1_macro"] for w, v in t.get("per_week", {}).items()},
            "dataset": m.get("dataset"),
            "metrics_source": "measured",
        })
    except (OSError, KeyError, ValueError) as e:
        # Better to say the numbers are missing than to show stale ones.
        info["metrics_source"] = "unavailable"
        info["metrics_error"] = f"run ensemble_weekly.py ({e})"

    return jsonify(info)


if __name__ == "__main__":
    print("=" * 60)
    print("Emotional Burnout NLP — Ensemble Inference Service")
    print("BERT + RoBERTa, soft voting")
    print("=" * 60)

    if not load_model():
        print("\n[X] Startup aborted — models could not be loaded.")
        sys.exit(1)

    print()
    print(f"  Base URL: http://localhost:{PORT}")
    print("  POST /api/analyze     {\"text\": \"...\"}")
    print("  GET  /api/health")
    print("  GET  /api/model-info")
    print()
    print("  Ctrl+C to stop")
    print("=" * 60)
    print()

    # debug=False: the reloader would load the 418 MB model twice
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
