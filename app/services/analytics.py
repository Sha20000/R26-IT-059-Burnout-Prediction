from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from meta_xai.opportunity_model import FEATURE_COLUMNS


def _curve_points(values_x: Any, values_y: Any, limit: int = 31) -> list[dict[str, float]]:
    points = list(zip(values_x, values_y))
    step = max(1, len(points) // limit)
    selected = points[::step]
    if points and selected[-1] != points[-1]:
        selected.append(points[-1])
    return [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in selected]


def _evaluation_curves(root: Path) -> dict[str, list[dict[str, float]]]:
    training_path = root / "data" / "training" / "intervention_outcomes.csv"
    model_path = root / "models" / "intervention_opportunity_model.joblib"
    if not training_path.exists() or not model_path.exists():
        return {"roc": [], "pr": []}
    frame = pd.read_csv(training_path)
    if "risk_change" not in frame.columns:
        frame["risk_change"] = frame["week17_risk"] - frame["week4_risk"]
    if "recent_risk_change" not in frame.columns:
        frame["recent_risk_change"] = frame["week17_risk"] - frame["week12_risk"]
    split = max(1, int(len(frame) * 0.8))
    split = min(split, len(frame) - 1)
    test = frame.iloc[split:]
    probabilities = joblib.load(model_path).predict_proba(test[FEATURE_COLUMNS])[:, 1]
    fpr, tpr, _ = roc_curve(test["intervention_success"], probabilities)
    precision, recall, _ = precision_recall_curve(test["intervention_success"], probabilities)
    return {"roc": _curve_points(fpr, tpr), "pr": _curve_points(recall, precision)}

def build_analytics(records: list[dict], manifest_path: str | Path, metrics_path: str | Path) -> dict[str, Any]:
    root = Path(metrics_path).parents[1]
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8")) if Path(metrics_path).exists() else None
    priority = {level: sum(row.get("priority_level") == level for row in records) for level in ("P1", "P2", "P3")}
    trajectories: dict[str, int] = {}
    opportunity: dict[str, int] = {band: 0 for band in ("high", "medium", "low")}
    for row in records:
        trajectories[row["trajectory"]] = trajectories.get(row["trajectory"], 0) + 1
        band = row.get("intervention_opportunity", {}).get("opportunity_band")
        if band in opportunity:
            opportunity[band] += 1
    return {
        "dataset": {
            "version": manifest.get("dataset_version"),
            "mode": manifest.get("data_mode"),
            "cohort": manifest.get("cohort_name"),
            "cohort_id": manifest.get("cohort_id"),
            "students": len(records),
            "observation_weeks": manifest.get("observation_period", {}).get("weeks", []),
        },
        "fusion": {
            "source_coverage": {"academic": 100, "behavioural": 100, "emotional": 100},
            "priority": priority,
            "trajectories": trajectories,
            "opportunity_bands": opportunity,
        },
        "upstream_models": manifest.get("reported_upstream_metrics", {}),
        "intervention_model": metrics or {"status": "not_trained"},
        "curves": _evaluation_curves(root),
    }