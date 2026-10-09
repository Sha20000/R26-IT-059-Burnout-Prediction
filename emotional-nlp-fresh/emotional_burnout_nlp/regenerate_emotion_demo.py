"""
Re-scores the Emotional Analysis demo cohort (my-academic-gru/burnout-demo/emotion-demo-data.js)
with the current serve_model.py ensemble, so the stored demo results match the live API.
IT22196392 — Induwara K.P.Y. | R26-IT-059

Texts are not changed; only each week's "result" and the latestResult / worstResult
summaries are rebuilt, using the same rules as emotion-app.js.

Run:  .venv/Scripts/python.exe regenerate_emotion_demo.py
"""
import json
from pathlib import Path

import serve_model

DEMO_JS = Path(__file__).resolve().parents[2] / "my-academic-gru" / "burnout-demo" / "emotion-demo-data.js"
PREFIX = "window.EMOTION_DEMO_STUDENTS = "
RISK_SCORE = {"Low": 1, "Medium": 3, "High": 7, "Critical": 10, "Unknown": 0}   # as in emotion-tab.js
STORED_KEYS = ["attention", "confidence", "prediction", "probabilities", "risk_level", "stress_score"]


def main():
    src = DEMO_JS.read_text(encoding="utf-8")
    start = src.index(PREFIX)
    header, body = src[:start], src[start + len(PREFIX):src.rindex(";")]
    students = json.loads(body)
    assert serve_model.load_model()

    for s in students:
        weeks = sorted(s["weeks"], key=int)
        for w in weeks:
            r = serve_model.analyze_text(s["weeks"][w]["text"])
            r["probabilities"] = dict(sorted(r["probabilities"].items()))
            old = s["weeks"][w]["result"]["prediction"]
            s["weeks"][w]["result"] = {k: r[k] for k in STORED_KEYS}
            print(f"{s['name']:20s} week {w:>2s}: {old:12s} -> {r['prediction']:12s} ({r['confidence']:.2f})")
        results = [s["weeks"][w]["result"] for w in weeks]
        s["latestResult"] = results[-1]
        # First week with the highest risk level (stable sort, same as emotion-app.js)
        s["worstResult"] = sorted(results, key=lambda r: -RISK_SCORE.get(r["risk_level"], 0))[0]

    header = header.replace(
        "real output from the fine-tuned BERT model\n * (serve_model.py)",
        "real output from the BERT + RoBERTa ensemble\n * (serve_model.py, weekly_v2 models)")
    header = header.replace(
        "Regenerate by re-running those texts against POST :5004/api/analyze.",
        "Regenerate with emotional_burnout_nlp/regenerate_emotion_demo.py.")
    DEMO_JS.write_text(header + PREFIX + json.dumps(students, indent=1, ensure_ascii=False) + ";\n",
                       encoding="utf-8")
    print(f"\nSaved {DEMO_JS}")


if __name__ == "__main__":
    main()
