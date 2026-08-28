#!/usr/bin/env python3
"""
Emotional Burnout NLP — warm inference service
IT22196392 — Induwara K.P.Y. | R26-IT-059

Loads the fine-tuned BERT model ONCE at startup and keeps it in memory, instead
of re-loading 418 MB of weights on every request the way predict.py does when
spawned per-call. Scoring logic is identical to predict.py, so results match.

Serves the integrated dashboard (Emotional Analysis tab) on port 5004,
alongside the other members' APIs:
    5001  academic / burnout        (my-academic-gru)
    5002  meta-integration
    5003  behavioural anomaly VAE
    5004  emotional NLP             <- this service
"""

import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)
CORS(app)  # dashboard is served from a different origin (:8000)

MODEL_PATH = Path(__file__).parent / "models/saved/bert_burnout"
PORT = 5004

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

TOKENIZER = None
MODEL = None


def load_model():
    """Load tokenizer + model once into module globals."""
    global TOKENIZER, MODEL

    if not MODEL_PATH.exists():
        print(f"[X] Model not found at {MODEL_PATH}")
        return False

    print(f"[..] Loading model from {MODEL_PATH}")
    print("     (one-time, ~20s — every request after this is instant)")
    try:
        TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        MODEL = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH), output_attentions=True)
        MODEL.eval()
    except Exception as e:
        print(f"[X] Failed to load model: {e}")
        return False

    labels = list(MODEL.config.id2label.values())
    print(f"[OK] Model loaded — {len(labels)} classes: {', '.join(labels)}")
    return True


def analyze_text(text):
    """Run inference. Returns the same dict shape predict.py prints."""
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = MODEL(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    predicted_id = probs.argmax().item()
    predicted_label = MODEL.config.id2label[predicted_id]
    confidence = float(probs.max().item())

    probabilities = {
        MODEL.config.id2label[i]: round(float(p), 4)
        for i, p in enumerate(probs)
    }

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

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
        "risk_level": RISK_MAP.get(predicted_label, "Unknown"),
        "stress_score": stress_score,
        "probabilities": probabilities,
        "attention": attention,
    }


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online" if MODEL is not None else "model_not_loaded",
        "service": "emotional-nlp",
        "model": "BERT fine-tuned + RoBERTa ensemble",
        "classes": list(MODEL.config.id2label.values()) if MODEL else [],
        "api_version": "1.0",
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
    """Static metrics shown in the dashboard's model performance panel."""
    return jsonify({
        "model": "BERT Fine-tuned + RoBERTa Ensemble",
        "accuracy": "82.32%",
        "f1_score": "82.32%",
        "training_samples": 51055,
        "num_classes": 7,
        "classes": list(MODEL.config.id2label.values()) if MODEL else [],
        "author": "IT22196392 — Induwara K.P.Y.",
        "project": "R26-IT-059",
        # Per-class F1, as shown in the original frontend's results table
        "per_class_f1": {
            "Normal": "92%", "Stress": "75%", "Anxiety": "82%",
            "Depression": "85%", "Bipolar": "82%",
            "Personality disorder": "77%", "Suicidal": "86%",
        },
    })


if __name__ == "__main__":
    print("=" * 60)
    print("Emotional Burnout NLP — Inference Service")
    print("=" * 60)

    if not load_model():
        print("\n[X] Startup aborted — model could not be loaded.")
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
