from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from .ingestion import (
    load_academic,
    load_behaviour,
    load_emotional,
    load_identity_mapping,
)

FEATURE_COLUMNS = [
    "week4_risk",
    "week8_risk",
    "week12_risk",
    "week17_risk",
    "risk_change",
    "recent_risk_change",
    "behavior_risk",
    "compliance_mean",
    "anomaly_mean",
    "high_anomaly_weeks",
    "compliance_trend",
    "emotional_stress",
    "evidence_coverage",
]

TARGET_COLUMN = "intervention_success"

FEATURE_METADATA = {
    "week4_risk": {"label": "Week 4 Academic Risk", "unit": "score"},
    "week8_risk": {"label": "Week 8 Academic Risk", "unit": "score"},
    "week12_risk": {"label": "Week 12 Academic Risk", "unit": "score"},
    "week17_risk": {"label": "Week 17 Academic Risk", "unit": "score"},
    "risk_change": {"label": "Overall Academic Trajectory Delta", "unit": "delta"},
    "recent_risk_change": {"label": "Recent Risk Velocity (W12-W17)", "unit": "delta"},
    "behavior_risk": {"label": "Mean Behavioral Risk Score", "unit": "score"},
    "compliance_mean": {"label": "Curriculum Compliance Mean", "unit": "score"},
    "anomaly_mean": {"label": "VAE Anomaly Score", "unit": "score"},
    "high_anomaly_weeks": {"label": "High Anomaly Weeks Triggered", "unit": "count"},
    "compliance_trend": {"label": "Curriculum Compliance Trend", "unit": "delta"},
    "emotional_stress": {"label": "Emotional Stress (NLP RoBERTa)", "unit": "score (0-4)"},
    "evidence_coverage": {"label": "Multi-Modal Evidence Coverage", "unit": "ratio"},
}


@dataclass(frozen=True)
class TrainingResult:
    rows: int
    train_rows: int
    test_rows: int
    positive_rate: float
    roc_auc: float
    average_precision: float
    f1: float
    artifact: str
    metrics: str


def bootstrap_intervention_outcomes(
    incoming_dir: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Synthesize the intervention_success proxy label:
    1 (Success) if week17_risk < week8_risk AND curriculum_compliance trend > 0.
    Otherwise 0.
    """
    incoming = Path(incoming_dir)
    academic_table = load_academic(incoming / "academic_predictions.csv")
    behaviour_table = load_behaviour(incoming / "behavior_predictions.csv")
    emotional_table = load_emotional(incoming / "emotional_predictions.csv")
    mapping_df, _ = load_identity_mapping(incoming / "student_mapping.csv")

    if mapping_df is None:
        raise ValueError("Could not load student_mapping.csv for bootstrapping")

    acad_map = {str(r["student_id"]): r for r in academic_table.records}
    beh_map = {str(r["student_id"]): r for r in behaviour_table.records}
    emo_map = {str(r["student_id"]): r for r in emotional_table.records}

    rows: list[dict[str, Any]] = []
    for _, row in mapping_df.iterrows():
        a = acad_map.get(str(row["academic_student_id"]))
        b = beh_map.get(str(row["behavior_student_id"]))
        e = emo_map.get(str(row["emotional_student_id"]))
        if not (a and b and e):
            continue

        w4 = float(a["week4_risk"])
        w8 = float(a["week8_risk"])
        w12 = float(a["week12_risk"])
        w17 = float(a["week17_risk"])
        comp_trend = float(b.get("compliance_trend", 0.0))

        # Target Bootstrapping Logic:
        # Academic risk decreased from week 8 to 17 AND curriculum compliance trend is positive
        is_success = 1 if (w17 < w8 and comp_trend > 0) else 0

        rows.append({
            "student_id": str(row["canonical_student_id"]),
            "as_of_week": 17,
            "week4_risk": round(w4, 4),
            "week8_risk": round(w8, 4),
            "week12_risk": round(w12, 4),
            "week17_risk": round(w17, 4),
            "risk_change": round(w17 - w4, 4),
            "recent_risk_change": round(w17 - w12, 4),
            "behavior_risk": round(float(b.get("behavior_risk_mean", 0.0)), 4),
            "compliance_mean": round(float(b.get("compliance_mean", 0.0)), 4),
            "anomaly_mean": round(float(b.get("anomaly_mean", 0.0)), 4),
            "high_anomaly_weeks": float(b.get("high_anomaly_weeks", 0)),
            "compliance_trend": round(comp_trend, 4),
            "emotional_stress": round(float(e.get("emotional_stress_score", 0.0)), 4),
            "evidence_coverage": 1.0,
            "programme": str(a.get("programme", "Computing")),
            TARGET_COLUMN: is_success,
        })

    frame = pd.DataFrame(rows)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
    return frame


def train(input_path: str | Path, output_dir: str | Path) -> TrainingResult:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    if "risk_change" not in frame.columns and "week17_risk" in frame.columns:
        frame["risk_change"] = frame["week17_risk"] - frame["week4_risk"]
    if "recent_risk_change" not in frame.columns and "week17_risk" in frame.columns:
        frame["recent_risk_change"] = frame["week17_risk"] - frame["week12_risk"]

    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(int)
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1")

    frame = frame.sort_values("as_of_week") if "as_of_week" in frame.columns else frame
    split = max(1, int(len(frame) * 0.8))
    if split >= len(frame):
        split = len(frame) - 1

    train_frame, test_frame = frame.iloc[:split], frame.iloc[split:]
    if train_frame[TARGET_COLUMN].nunique() < 2 or test_frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Train/test split must contain both target classes")

    y_train = train_frame[TARGET_COLUMN]
    y_test = test_frame[TARGET_COLUMN]
    pos_count = max(1, int(y_train.sum()))
    neg_count = len(y_train) - pos_count
    scale_pos_weight = float(neg_count / pos_count)

    model = XGBClassifier(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(train_frame[FEATURE_COLUMNS], y_train)

    probabilities = model.predict_proba(test_frame[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model_type": "XGBoost Classifier + TreeSHAP",
        "target": TARGET_COLUMN,
        "rows": len(frame),
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "positive_rate": round(float(frame[TARGET_COLUMN].mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "feature_columns": FEATURE_COLUMNS,
        "split": "time ordered 80/20",
    }

    artifact = output_dir / "intervention_opportunity_model.joblib"
    metrics_path = output_dir / "intervention_opportunity_metrics.json"

    joblib.dump(model, artifact)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainingResult(
        artifact=str(artifact),
        metrics=str(metrics_path),
        **{key: metrics[key] for key in ["rows", "train_rows", "test_rows", "positive_rate", "roc_auc", "average_precision", "f1"]},
    )


def explain_prediction(model: Any, record: dict | pd.DataFrame) -> dict:
    """
    Wrap XGBoost prediction with shap.TreeExplainer to compute feature contributions
    and format them into mathematically sound natural language explanations.
    """
    if isinstance(record, dict):
        frame = pd.DataFrame([record])
    else:
        frame = record.copy()

    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0

    frame = frame[FEATURE_COLUMNS].astype(float)
    probability = float(model.predict_proba(frame)[0, 1])

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(frame)[0]

    abs_sum = float(np.sum(np.abs(shap_vals)))
    contributions = []

    for feat, s_val in zip(FEATURE_COLUMNS, shap_vals):
        val = float(frame.iloc[0][feat])
        pct = (float(s_val) / abs_sum) * 100.0 if abs_sum > 0 else 0.0
        meta = FEATURE_METADATA.get(feat, {"label": feat, "unit": ""})

        # Formulate human readable narrative
        sign_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        direction = "positive" if pct >= 0 else "negative"

        if feat == "emotional_stress":
            stress_level = "Severe" if val >= 3.0 else "Elevated" if val >= 2.0 else "Mild" if val >= 1.0 else "Low"
            narrative = f"{sign_str} priority: Emotional stress ({stress_level}: {val:.1f}/4.0)"
        elif feat == "compliance_trend":
            trend_desc = "improving" if val > 0 else "declining"
            narrative = f"{sign_str} responsiveness: Curriculum compliance {trend_desc} (delta {val:+.2f})"
        elif feat == "compliance_mean":
            narrative = f"{sign_str} stability: Mean curriculum compliance at {val*100:.0f}%"
        elif feat == "anomaly_mean":
            anomaly_desc = "erratic login activity" if val >= 0.5 else "regular behavioral patterns"
            narrative = f"{sign_str} anomaly: VAE detected {anomaly_desc} ({val:.2f})"
        elif feat in {"week8_risk", "week17_risk"}:
            narrative = f"{sign_str} trajectory: Academic risk at {meta['label']} is {val:.2f}"
        else:
            narrative = f"{sign_str} contribution from {meta['label']} ({val:.2f})"

        contributions.append({
            "feature": feat,
            "display_name": meta["label"],
            "value": round(val, 3),
            "shap_value": round(float(s_val), 4),
            "percentage": round(pct, 1),
            "direction": direction,
            "narrative": narrative,
        })

    # Sort contributions by absolute percentage impact
    contributions.sort(key=lambda x: abs(x["percentage"]), reverse=True)

    # Synthesize concise clinical 'Why' summary from top drivers
    top_drivers = contributions[:3]
    top_driver_summaries = [c["narrative"] for c in top_drivers if abs(c["percentage"]) >= 5.0]
    if not top_driver_summaries:
        top_driver_summaries = [c["narrative"] for c in top_drivers[:1]]

    why_summary = " | ".join(top_driver_summaries)

    return {
        "intervention_success_probability": round(probability, 4),
        "opportunity_band": "high" if probability >= 0.65 else "medium" if probability >= 0.4 else "low",
        "model": "intervention-opportunity-xgboost-v2",
        "feature_contributions": contributions,
        "top_drivers": top_drivers,
        "why_explanation": why_summary,
    }


def predict(model_path: str | Path, record: dict) -> dict:
    model = joblib.load(model_path)
    return explain_prediction(model, record)
