from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "week4_risk", "week8_risk", "week12_risk", "week17_risk",
    "risk_change", "recent_risk_change", "behavior_risk",
    "emotional_stress", "evidence_coverage", "programme",
]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column != "programme"]
TARGET_COLUMN = "intervention_success"


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


def _pipeline() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    features = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, ["programme"]),
    ])
    return Pipeline([
        ("features", features),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def train(input_path: str | Path, output_dir: str | Path) -> TrainingResult:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path)
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
        raise ValueError("Temporal train/test split must contain both target classes")
    model = _pipeline()
    model.fit(train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN])
    probabilities = model.predict_proba(test_frame[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "target": TARGET_COLUMN,
        "rows": len(frame),
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "positive_rate": round(float(frame[TARGET_COLUMN].mean()), 4),
        "roc_auc": round(float(roc_auc_score(test_frame[TARGET_COLUMN], probabilities)), 4),
        "average_precision": round(float(average_precision_score(test_frame[TARGET_COLUMN], probabilities)), 4),
        "f1": round(float(f1_score(test_frame[TARGET_COLUMN], predictions)), 4),
        "feature_columns": FEATURE_COLUMNS,
        "split": "time ordered 80/20",
    }
    artifact = output_dir / "intervention_opportunity_model.joblib"
    metrics_path = output_dir / "intervention_opportunity_metrics.json"
    joblib.dump(model, artifact)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return TrainingResult(artifact=str(artifact), metrics=str(metrics_path), **{key: metrics[key] for key in ["rows", "train_rows", "test_rows", "positive_rate", "roc_auc", "average_precision", "f1"]})


def predict(model_path: str | Path, record: dict) -> dict:
    model = joblib.load(model_path)
    frame = pd.DataFrame([record], columns=FEATURE_COLUMNS)
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        "intervention_success_probability": round(probability, 4),
        "opportunity_band": "high" if probability >= 0.65 else "medium" if probability >= 0.4 else "low",
        "model": "intervention-opportunity-logistic-v1",
    }
