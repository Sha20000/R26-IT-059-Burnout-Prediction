import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = Path(__file__).parent / "models/saved/bert_burnout"

RISK_MAP = {
    "Normal": "Low",
    "Stress": "Medium",
    "Anxiety": "Medium",
    "Depression": "High",
    "Bipolar": "High",
    "Personality disorder": "High",
    "Suicidal": "Critical",
}

def predict(text):
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH), output_attentions=True)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    predicted_id = probs.argmax().item()
    predicted_label = model.config.id2label[predicted_id]
    confidence = float(probs.max().item())

    probabilities = {
        model.config.id2label[i]: round(float(p), 4)
        for i, p in enumerate(probs)
    }

    stress_score = round(float(
        probabilities.get("Suicidal", 0)             * 1.0 +
        probabilities.get("Depression", 0)            * 0.8 +
        probabilities.get("Bipolar", 0)               * 0.75 +
        probabilities.get("Personality disorder", 0)  * 0.7 +
        probabilities.get("Anxiety", 0)               * 0.6 +
        probabilities.get("Stress", 0)                * 0.5
    ) * 100, 1)

    all_attentions = torch.stack(outputs.attentions)
    avg_attention = all_attentions.mean(dim=0).mean(dim=1).squeeze()
    cls_attention = avg_attention[0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    clean_tokens = tokens[1:-1]
    clean_weights = cls_attention[1:-1]
    total = clean_weights.sum()
    if total > 0:
        clean_weights = clean_weights / total

    attention = [
        {"token": t[2:] if t.startswith("##") else t, "weight": round(float(w), 6)}
        for t, w in zip(clean_tokens, clean_weights)
    ]

    result = {
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
        "risk_level": RISK_MAP.get(predicted_label, "Unknown"),
        "stress_score": stress_score,
        "probabilities": probabilities,
        "attention": attention,
    }

    print(json.dumps(result))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No text provided"}))
        sys.exit(1)
    text = sys.argv[1]
    if not text.strip() or len(text.strip().split()) < 3:
        print(json.dumps({"error": "Please enter at least a few words"}))
        sys.exit(1)
    predict(text)
