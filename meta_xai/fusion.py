from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .contracts import SourceIssue
from .ingestion import (
    build_alignment_report,
    load_academic,
    load_behaviour,
    load_emotional,
    load_identity_mapping,
    report_as_dict,
)
from .opportunity_model import FEATURE_COLUMNS
from .reasoning import calculate_priority, classify_academic_trajectory

RECOMMENDED_ACTIONS = {
    "rapidly_escalating": "Academic advisor check-in",
    "persistent_high": "Programme review",
    "improving": "Recognition and light-touch follow-up",
    "stable_low": "Routine monitoring",
    "late_emerging": "Academic advisor review",
    "unstable": "Human review of conflicting evidence",
}


def _source_records(table: Any) -> dict[str, dict]:
    return {str(record["student_id"]): record for record in table.records}


def _validate_cohort(mapping: pd.DataFrame, manifest_path: Path | None) -> list[SourceIssue]:
    issues: list[SourceIssue] = []
    if "cohort_id" not in mapping.columns:
        return [SourceIssue("identity", "missing_cohort_id", "Mapping must include cohort_id")]
    cohorts = {str(value).strip() for value in mapping["cohort_id"] if str(value).strip()}
    if len(cohorts) != 1:
        issues.append(SourceIssue("identity", "inconsistent_cohort", "All mapping rows must use one cohort_id"))
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = str(manifest.get("cohort_id", "")).strip()
        if expected and cohorts != {expected}:
            issues.append(SourceIssue("identity", "cohort_mismatch", "Mapping cohort_id does not match manifest cohort_id"))
    return issues


def build_evidence_profiles(
    input_dir: str | Path,
    mapping_path: str | Path,
    manifest_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> tuple[list[dict], dict]:
    input_dir = Path(input_dir)
    mapping_path = Path(mapping_path)
    manifest_path = Path(manifest_path) if manifest_path else input_dir / "manifest.json"
    sources = [
        load_academic(input_dir / "academic_predictions.csv"),
        load_behaviour(input_dir / "behavior_predictions.csv"),
        load_emotional(input_dir / "emotional_predictions.csv"),
    ]
    mapping, mapping_issues = load_identity_mapping(mapping_path)
    if mapping is not None:
        mapping_issues.extend(_validate_cohort(mapping, manifest_path))
    report = build_alignment_report(sources, mapping, mapping_issues)
    report_dict = report_as_dict(report)
    if not report.is_ready_for_fusion or mapping is None:
        raise ValueError(json.dumps({"message": "Inputs are not ready for fusion", "alignment": report_dict}))

    academic = _source_records(sources[0])
    behaviour = _source_records(sources[1])
    emotional = _source_records(sources[2])
    opportunity_model = joblib.load(model_path) if model_path and Path(model_path).exists() else None
    opportunity_records: list[dict] = []
    profiles: list[dict] = []
    for row in mapping.to_dict(orient="records"):
        source_rows = {
            "academic": academic.get(str(row["academic_student_id"])),
            "behaviour": behaviour.get(str(row["behavior_student_id"])),
            "emotional": emotional.get(str(row["emotional_student_id"])),
        }
        available = sum(value is not None for value in source_rows.values())
        if available != 3:
            continue
        academic_row = source_rows["academic"]
        behaviour_row = source_rows["behaviour"]
        emotional_row = source_rows["emotional"]
        risks = [float(academic_row[f"week{week}_risk"]) for week in (4, 8, 12, 17)]
        trajectory = classify_academic_trajectory(*risks)
        coverage = available / 3
        priority = calculate_priority(trajectory, 0.7, coverage, 0.7, available, 3)
        profile = {
            "student_id": str(row["canonical_student_id"]),
            "student_name": f"Student {str(row['canonical_student_id']).split('_')[-1]}",
            "canonical_student_id": str(row["canonical_student_id"]),
            "academic_student_id": str(row["academic_student_id"]),
            "behavior_student_id": str(row["behavior_student_id"]),
            "emotional_student_id": str(row["emotional_student_id"]),
            "cohort_id": str(row["cohort_id"]),
            "programme": str(academic_row.get("programme", "OULAD cohort")),
            "academic_risks": risks,
            "academic_risk": float(academic_row["academic_risk"]),
            "behavior_risk": float(behaviour_row["behavior_risk_mean"]),
            "behavior_risk_max": float(behaviour_row["behavior_risk_max"]),
            "behavior_risk_change": float(behaviour_row.get("behavior_risk_change", 0.0)),
            "emotional_stress": float(emotional_row["emotional_stress_score"]),
            "evidence_coverage": coverage,
            "available_sources": available,
            "expected_sources": 3,
            "trajectory": trajectory.state,
            "trajectory_explanation": trajectory.explanation,
            "priority_score": priority.score,
            "priority_level": priority.level,
            "confidence": priority.confidence,
            "reason_codes": list(priority.reason_codes),
            "recommended_action": RECOMMENDED_ACTIONS[trajectory.state],
            "owner_role": "Programme advisor" if priority.level == "P1" else "Student success coach",
            "due_date": str(date.today() + timedelta(days=3 if priority.level == "P1" else 7)),
            "status": "New",
            "follow_up": "Pending",
            "model_evidence": {
                "academic": f"Week 4 {risks[0]:.2f} -> Week 17 {risks[3]:.2f}",
                "behavioural": f"Mean weekly behavioural risk {float(behaviour_row['behavior_risk_mean']):.2f}",
                "emotional": f"Stress signal {float(emotional_row['emotional_stress_score']):.0f}/4",
            },
            "data_mode": "synthetic_demo",
        }
        if opportunity_model is not None:
            opportunity_records.append({
                "week4_risk": risks[0], "week8_risk": risks[1], "week12_risk": risks[2], "week17_risk": risks[3],
                "risk_change": trajectory.change, "recent_risk_change": risks[3] - risks[2],
                "behavior_risk": profile["behavior_risk"], "emotional_stress": profile["emotional_stress"],
                "evidence_coverage": coverage, "programme": str(academic_row.get("programme", "Unknown")),
            })
        else:
            profile["intervention_opportunity"] = {"status": "not_trained"}
        profiles.append(profile)
    if opportunity_model is not None:
        probabilities = opportunity_model.predict_proba(pd.DataFrame(opportunity_records, columns=FEATURE_COLUMNS))[:, 1]
        for profile, probability in zip(profiles, probabilities):
            profile["intervention_opportunity"] = {
                "intervention_success_probability": round(float(probability), 4),
                "opportunity_band": "high" if probability >= 0.65 else "medium" if probability >= 0.4 else "low",
                "model": "intervention-opportunity-logistic-v1",
            }
    return profiles, report_dict


def write_profiles_csv(profiles: list[dict], output_path: str | Path) -> None:
    frame = pd.DataFrame(profiles)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
