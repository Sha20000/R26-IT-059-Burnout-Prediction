"""
Builds the test-set student cohort shown in the dashboard's Emotional Analysis tab
(my-academic-gru/burnout-demo/emotion-dataset-students.js).
IT22196392 — Induwara K.P.Y. | R26-IT-059

Students come from the weekly_v2 *test* split only (never seen in training). Each of
their four weekly texts is scored with the same ensemble the live API serves
(serve_model.analyze_text), so the stored predictions and attention heatmaps are real
model output. The dataset label is kept next to each prediction for comparison.

Selection: students whose four texts are all <= MAX_WORDS words (keeps the page light),
grouped by the shape of their predicted risk across the semester (worsening, improving,
stable low, stable high, fluctuating) and sampled evenly from each group (seed 42).

Run:  .venv/Scripts/python.exe build_dashboard_cohort.py
"""
import json
from pathlib import Path

import pandas as pd

import serve_model
from preprocess_weekly import OUT_DIR as DATA_DIR

BASE_DIR = Path(__file__).parent
OUT_JS = BASE_DIR.parents[1] / "my-academic-gru" / "burnout-demo" / "emotion-dataset-students.js"
WEEKLY_SCORES = BASE_DIR / "results" / "weekly_v2_emotional_stress_weekly.csv"
PER_GROUP = 8
MAX_WORDS = 250
SEED = 42
RISK_SCORE = {"Low": 1, "Medium": 3, "High": 7, "Critical": 10}   # as in emotion-tab.js
STORED_KEYS = ["attention", "confidence", "prediction", "probabilities", "risk_level", "stress_score"]


def trajectory(risks):
    s = [RISK_SCORE[r] for r in risks]
    if all(r == "Low" for r in risks):
        return "stable low"
    if all(v >= 7 for v in s):
        return "stable high"
    if s[-1] > s[0]:
        return "worsening"
    if s[-1] < s[0]:
        return "improving"
    return "fluctuating"


def main():
    rows = pd.read_csv(DATA_DIR / "weekly_clean.csv", dtype={"student_id": str})
    test = rows[rows.split == "test"].copy()
    scores = pd.read_csv(WEEKLY_SCORES, dtype={"student_id": str})
    test = test.merge(scores[["student_id", "week", "risk_level"]], on=["student_id", "week"])

    per_student = test.groupby("student_id").agg(
        n=("week", "size"), max_words=("word_count", "max"),
        risks=("risk_level", list)).reset_index()
    pool = per_student[(per_student.n == 4) & (per_student.max_words <= MAX_WORDS)].copy()
    pool["group"] = pool["risks"].map(trajectory)
    picked = pd.concat([g.sample(min(PER_GROUP, len(g)), random_state=SEED)
                        for _, g in pool.groupby("group")])
    print(f"[1] {len(pool)} eligible test students -> picked {len(picked)}:",
          picked.group.value_counts().to_dict())

    assert serve_model.load_model()
    students = []
    for sid in sorted(picked.student_id):
        weeks = {}
        for _, r in test[test.student_id == sid].sort_values("week").iterrows():
            res = serve_model.analyze_text(r["text_clean"])
            res["probabilities"] = dict(sorted(res["probabilities"].items()))
            weeks[str(int(r["week"]))] = {"text": r["text_clean"], "label": r["status"],
                                          "result": {k: res[k] for k in STORED_KEYS}}
        results = [weeks[w]["result"] for w in sorted(weeks, key=int)]
        students.append({
            "id": sid, "name": sid, "dataset": True, "weekNums": [2, 4, 8, 12],
            "weeks": weeks,
            "latestResult": results[-1],
            # First week with the highest risk level (min keeps the first tie, same as emotion-app.js)
            "worstResult": min(results, key=lambda x: -RISK_SCORE.get(x["risk_level"], 0)),
        })

    preds = [(w["result"]["prediction"], w["label"]) for s in students for w in s["weeks"].values()]
    acc = sum(p == l for p, l in preds) / len(preds)
    header = (
        "/* ============================================================================\n"
        " * Test-set student cohort — Emotional Analysis tab\n"
        " * IT22196392 — Induwara K.P.Y. | R26-IT-059\n"
        " *\n"
        f" * {len(students)} students from the weekly_v2 test split (not used in training), weeks 2/4/8/12.\n"
        " * Every prediction, probability and attention weight is real output of the\n"
        " * BERT + RoBERTa ensemble (serve_model.py); \"label\" is the dataset label.\n"
        f" * Agreement with the dataset label on these {len(preds)} texts: {acc:.1%}.\n"
        " * Regenerate with emotional_burnout_nlp/build_dashboard_cohort.py.\n"
        " * ========================================================================== */\n\n"
    )
    OUT_JS.write_text(header + "window.EMOTION_DATASET_STUDENTS = "
                      + json.dumps(students, ensure_ascii=False, separators=(",", ":")) + ";\n",
                      encoding="utf-8")
    print(f"[2] {len(students)} students, {len(preds)} texts, label agreement {acc:.1%}")
    print(f"[3] Saved {OUT_JS} ({OUT_JS.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
