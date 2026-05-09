"""Train an engagement classification model using OULAD data."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

STUDENT_VLE_PATH = DATA_DIR / "studentVle.csv"
STUDENT_INFO_PATH = DATA_DIR / "studentInfo.csv"
ASSESSMENTS_PATH = DATA_DIR / "assessments.csv"

OUTPUT_PREDICTIONS = DATA_DIR / "engagement_predictions.csv"
OUTPUT_MODEL = MODEL_DIR / "engagement_model.joblib"
OUTPUT_REPORT = MODEL_DIR / "engagement_report.json"


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the required OULAD datasets."""
    student_vle = pd.read_csv(STUDENT_VLE_PATH)
    student_info = pd.read_csv(STUDENT_INFO_PATH)
    assessments = pd.read_csv(ASSESSMENTS_PATH)
    return student_vle, student_info, assessments


def build_engagement_features(
    student_vle: pd.DataFrame, student_info: pd.DataFrame, assessments: pd.DataFrame
) -> pd.DataFrame:
    """Create engagement features per student."""
    # Normalize column names used across OULAD tables.
    student_vle = student_vle.rename(columns={"id_student": "student_id"})
    student_info = student_info.rename(columns={"id_student": "student_id"})

    # Aggregate click activity.
    vle_group = student_vle.groupby("student_id")
    total_clicks = vle_group["sum_click"].sum().rename("total_clicks")
    active_days = vle_group["date"].nunique().rename("active_days")
    min_day = vle_group["date"].min().rename("min_day")
    max_day = vle_group["date"].max().rename("max_day")

    # Compute inactivity gap as the largest gap between active days.
    def _max_gap(dates: pd.Series) -> int:
        days = np.sort(dates.unique())
        if len(days) <= 1:
            return 0
        gaps = np.diff(days)
        return int(gaps.max())

    inactivity_gap = vle_group["date"].apply(_max_gap).rename("inactivity_gap")

    # Merge in assessment interactions based on module/presentation membership.
    assessment_counts = (
        assessments.groupby(["code_module", "code_presentation"])["id_assessment"]
        .count()
        .rename("assessment_interactions")
        .reset_index()
    )

    student_meta = student_info[["student_id", "code_module", "code_presentation"]].copy()
    student_meta = student_meta.merge(
        assessment_counts, on=["code_module", "code_presentation"], how="left"
    )

    student_features = (
        pd.DataFrame({
            "student_id": total_clicks.index,
            "total_clicks": total_clicks.values,
            "active_days": active_days.values,
            "min_day": min_day.values,
            "max_day": max_day.values,
            "inactivity_gap": inactivity_gap.values,
        })
        .merge(student_meta, on="student_id", how="left")
        .drop(columns=["code_module", "code_presentation"], errors="ignore")
    )

    # Average clicks per active day.
    student_features["average_clicks"] = (
        student_features["total_clicks"] / student_features["active_days"].clip(lower=1)
    )

    # Click frequency across the total observed days window.
    observed_days = (student_features["max_day"] - student_features["min_day"] + 1).clip(
        lower=1
    )
    student_features["click_frequency"] = student_features["total_clicks"] / observed_days

    # Fill missing assessment counts with 0.
    student_features["assessment_interactions"] = student_features[
        "assessment_interactions"
    ].fillna(0)

    student_features = student_features.drop(columns=["min_day", "max_day"], errors="ignore")

    return student_features


def create_engagement_labels(features: pd.DataFrame) -> pd.DataFrame:
    """Assign LOW/MEDIUM/HIGH engagement labels using a rule-based strategy."""
    clicks_p33 = features["total_clicks"].quantile(0.33)
    clicks_p66 = features["total_clicks"].quantile(0.66)
    active_p33 = features["active_days"].quantile(0.33)
    active_p66 = features["active_days"].quantile(0.66)
    gap_p66 = features["inactivity_gap"].quantile(0.66)

    def _label(row: pd.Series) -> str:
        high_engagement = row["total_clicks"] >= clicks_p66 and row["active_days"] >= active_p66
        low_engagement = row["total_clicks"] <= clicks_p33 or row["active_days"] <= active_p33
        if row["inactivity_gap"] >= gap_p66:
            low_engagement = True

        if high_engagement:
            return "HIGH"
        if low_engagement:
            return "LOW"
        return "MEDIUM"

    labeled = features.copy()
    labeled["engagement_label"] = labeled.apply(_label, axis=1)
    return labeled


def train_models(features: pd.DataFrame) -> tuple[Pipeline, dict, LabelEncoder]:
    """Train Logistic Regression and Random Forest models."""
    feature_cols = [
        "total_clicks",
        "average_clicks",
        "active_days",
        "inactivity_gap",
        "assessment_interactions",
        "click_frequency",
    ]

    X = features[feature_cols].fillna(0)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(features["engagement_label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = feature_cols
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)], remainder="drop"
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    best_model = None
    best_score = -1

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )

        results[name] = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
        }

        if f1 > best_score:
            best_score = f1
            best_model = pipeline

    return best_model, results, label_encoder


def build_predictions(
    model: Pipeline, label_encoder: LabelEncoder, features: pd.DataFrame
) -> pd.DataFrame:
    """Generate engagement predictions for all students."""
    feature_cols = [
        "total_clicks",
        "average_clicks",
        "active_days",
        "inactivity_gap",
        "assessment_interactions",
        "click_frequency",
    ]

    X = features[feature_cols].fillna(0)
    proba = model.predict_proba(X)
    preds = model.predict(X)

    # Probability of the HIGH class is used as the engagement score.
    high_index = list(label_encoder.classes_).index("HIGH")
    engagement_score = proba[:, high_index]
    confidence = proba.max(axis=1)
    labels = label_encoder.inverse_transform(preds)

    return pd.DataFrame(
        {
            "student_id": features["student_id"],
            "engagement_score": engagement_score.round(4),
            "risk_level": labels,
            "confidence": confidence.round(4),
        }
    )


def save_outputs(model: Pipeline, report: dict, predictions: pd.DataFrame) -> None:
    """Persist model and predictions to disk."""
    joblib.dump(model, OUTPUT_MODEL)

    with OUTPUT_REPORT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    predictions.to_csv(OUTPUT_PREDICTIONS, index=False)


def main() -> None:
    """Run the full engagement modeling pipeline."""
    student_vle, student_info, assessments = load_datasets()
    features = build_engagement_features(student_vle, student_info, assessments)
    labeled = create_engagement_labels(features)

    model, report, label_encoder = train_models(labeled)
    predictions = build_predictions(model, label_encoder, labeled)

    save_outputs(model, report, predictions)

    print("Training complete. Best model saved to:", OUTPUT_MODEL)
    print("Metrics report saved to:", OUTPUT_REPORT)
    print("Predictions saved to:", OUTPUT_PREDICTIONS)


if __name__ == "__main__":
    main()
