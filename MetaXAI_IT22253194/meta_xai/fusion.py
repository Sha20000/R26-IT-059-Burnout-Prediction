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
    resolve_student_identity,
)
from .opportunity_model import FEATURE_COLUMNS, predict
from .reasoning import calculate_priority, classify_academic_trajectory, rank_triage_queue

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
        priority = calculate_priority(
            trajectory,
            actionability=0.7,
            evidence_confidence=coverage,
            opportunity_window=0.7,
            available_sources=available,
            expected_sources=3,
            emotional_stress=float(emotional_row["emotional_stress_score"]),
            high_anomaly_weeks=float(behaviour_row.get("high_anomaly_weeks", 0.0)),
            behavior_risk=float(behaviour_row["behavior_risk_mean"]),
        )
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
            "base_priority_level": priority.level,
            "effective_priority": priority.level,
            "confidence": priority.confidence,
            "is_multi_modal_override": priority.is_multi_modal_override,
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
            "compliance_mean": float(behaviour_row.get("compliance_mean", 0.0)),
            "anomaly_mean": float(behaviour_row.get("anomaly_mean", 0.0)),
            "high_anomaly_weeks": float(behaviour_row.get("high_anomaly_weeks", 0.0)),
            "compliance_trend": float(behaviour_row.get("compliance_trend", 0.0)),
        }
        if opportunity_model is not None:
            opportunity_records.append({
                "week4_risk": risks[0],
                "week8_risk": risks[1],
                "week12_risk": risks[2],
                "week17_risk": risks[3],
                "risk_change": trajectory.change,
                "recent_risk_change": risks[3] - risks[2],
                "behavior_risk": profile["behavior_risk"],
                "compliance_mean": profile["compliance_mean"],
                "anomaly_mean": profile["anomaly_mean"],
                "high_anomaly_weeks": profile["high_anomaly_weeks"],
                "compliance_trend": profile["compliance_trend"],
                "emotional_stress": profile["emotional_stress"],
                "evidence_coverage": coverage,
            })
        else:
            profile["intervention_opportunity"] = {"status": "not_trained"}
        profiles.append(profile)

    if opportunity_model is not None and opportunity_records:
        import numpy as np
        import shap
        from .opportunity_model import FEATURE_METADATA

        opp_frame = pd.DataFrame(opportunity_records, columns=FEATURE_COLUMNS)
        explainer = shap.TreeExplainer(opportunity_model)
        probabilities = opportunity_model.predict_proba(opp_frame)[:, 1]
        shap_values_batch = explainer.shap_values(opp_frame)

        for profile, prob, shap_vals, (_, row_data) in zip(profiles, probabilities, shap_values_batch, opp_frame.iterrows()):
            abs_sum = float(np.sum(np.abs(shap_vals)))
            contributions = []
            for feat, s_val in zip(FEATURE_COLUMNS, shap_vals):
                val = float(row_data[feat])
                pct = (float(s_val) / abs_sum) * 100.0 if abs_sum > 0 else 0.0
                meta = FEATURE_METADATA.get(feat, {"label": feat, "unit": ""})
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

            contributions.sort(key=lambda x: abs(x["percentage"]), reverse=True)
            top_drivers = contributions[:3]
            top_driver_summaries = [c["narrative"] for c in top_drivers if abs(c["percentage"]) >= 5.0]
            if not top_driver_summaries:
                top_driver_summaries = [c["narrative"] for c in top_drivers[:1]]

            profile["intervention_opportunity"] = {
                "intervention_success_probability": round(float(prob), 4),
                "opportunity_band": "high" if prob >= 0.65 else "medium" if prob >= 0.4 else "low",
                "model": "intervention-opportunity-xgboost-v2",
                "feature_contributions": contributions,
                "top_drivers": top_drivers,
                "why_explanation": " | ".join(top_driver_summaries),
            }
    ranked_profiles, queue_summary = rank_triage_queue(profiles, advisor_capacity=5)
    report_dict["queue_summary"] = queue_summary
    return ranked_profiles, report_dict


def write_profiles_csv(profiles: list[dict], output_path: str | Path) -> None:
    frame = pd.DataFrame(profiles)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def ingest_student_payload(
    payload: dict,
    mapping_df: pd.DataFrame | None = None,
    model_path: str | Path | None = None,
) -> dict:
    """
    Ingest a single streamed multi-modal student event:
    1. Resolves identities across OULAD_*, BEHAVIOR_*, and EMOTIONAL_* namespaces.
    2. Assembles multi-modal features and extracts trajectories.
    3. Executes multi-modal priority reasoning with overrides.
    4. Evaluates XGBoost intervention opportunity + TreeSHAP explanations.
    5. Returns a unified clinical student triage profile.
    """
    # 1. Identity Resolution
    raw_id = (
        payload.get("canonical_student_id")
        or payload.get("student_id")
        or payload.get("academic_student_id")
        or payload.get("behavior_student_id")
        or payload.get("emotional_student_id")
        or "UNKNOWN_STUDENT"
    )

    if mapping_df is not None:
        resolved = resolve_student_identity(str(raw_id), mapping_df)
        canonical_id = resolved["canonical_student_id"]
        academic_id = resolved["academic_student_id"]
        behavior_id = resolved["behavior_student_id"]
        emotional_id = resolved["emotional_student_id"]
        cohort_id = resolved.get("cohort_id", "OULAD_COHORT_2026_01")
    else:
        canonical_id = str(payload.get("canonical_student_id") or raw_id)
        academic_id = str(payload.get("academic_student_id") or raw_id)
        behavior_id = str(payload.get("behavior_student_id") or raw_id)
        emotional_id = str(payload.get("emotional_student_id") or raw_id)
        cohort_id = str(payload.get("cohort_id", "OULAD_COHORT_2026_01"))

    # 2. Extract multi-modal evidence components
    acad_data = payload.get("academic", {})
    beh_data = payload.get("behaviour", {})
    emo_data = payload.get("emotional", {})

    # Academic features
    if "academic_risks" in payload:
        risks = [float(v) for v in payload["academic_risks"]]
    elif "week4_risk" in acad_data:
        risks = [float(acad_data.get(f"week{w}_risk", 0.2)) for w in (4, 8, 12, 17)]
    elif "week4_risk" in payload:
        risks = [float(payload.get(f"week{w}_risk", 0.2)) for w in (4, 8, 12, 17)]
    else:
        risks = [0.2, 0.2, 0.2, 0.2]

    academic_risk = float(acad_data.get("academic_risk", payload.get("academic_risk", risks[-1])))
    programme = str(acad_data.get("programme", payload.get("programme", "Computing")))

    # Behavioral features
    behavior_risk = float(beh_data.get("behavior_risk_mean", payload.get("behavior_risk", 0.0)))
    behavior_risk_max = float(beh_data.get("behavior_risk_max", payload.get("behavior_risk_max", behavior_risk)))
    behavior_risk_change = float(beh_data.get("behavior_risk_change", payload.get("behavior_risk_change", 0.0)))
    compliance_mean = float(beh_data.get("compliance_mean", payload.get("compliance_mean", 0.8)))
    anomaly_mean = float(beh_data.get("anomaly_mean", payload.get("anomaly_mean", 0.2)))
    high_anomaly_weeks = float(beh_data.get("high_anomaly_weeks", payload.get("high_anomaly_weeks", 0.0)))
    compliance_trend = float(beh_data.get("compliance_trend", payload.get("compliance_trend", 0.0)))

    # Emotional features
    emotional_stress = float(emo_data.get("emotional_stress_score", payload.get("emotional_stress", 0.0)))

    # 3. Trajectory & Priority Reasoning
    trajectory = classify_academic_trajectory(*risks)
    coverage = 1.0
    priority = calculate_priority(
        trajectory,
        actionability=0.7,
        evidence_confidence=coverage,
        opportunity_window=0.7,
        available_sources=3,
        expected_sources=3,
        emotional_stress=emotional_stress,
        high_anomaly_weeks=high_anomaly_weeks,
        behavior_risk=behavior_risk,
    )

    if emotional_stress >= 3.0:
        recommended_action = "Immediate wellbeing and advisor check-in"
    elif high_anomaly_weeks >= 1:
        recommended_action = "Investigate study pattern and login anomalies"
    else:
        recommended_action = RECOMMENDED_ACTIONS.get(trajectory.state, "Advisor check-in")

    # 4. XGBoost & TreeSHAP Inference
    opp_result: dict[str, Any] = {"status": "not_trained"}
    if model_path and Path(model_path).exists():
        opp_feature_dict = {
            "week4_risk": risks[0],
            "week8_risk": risks[1],
            "week12_risk": risks[2],
            "week17_risk": risks[3],
            "risk_change": trajectory.change,
            "recent_risk_change": risks[3] - risks[2],
            "behavior_risk": behavior_risk,
            "compliance_mean": compliance_mean,
            "anomaly_mean": anomaly_mean,
            "high_anomaly_weeks": high_anomaly_weeks,
            "compliance_trend": compliance_trend,
            "emotional_stress": emotional_stress,
            "evidence_coverage": coverage,
        }
        opp_result = predict(model_path, opp_feature_dict)

    profile = {
        "student_id": canonical_id,
        "student_name": f"Student {canonical_id.split('_')[-1]}",
        "canonical_student_id": canonical_id,
        "academic_student_id": academic_id,
        "behavior_student_id": behavior_id,
        "emotional_student_id": emotional_id,
        "cohort_id": cohort_id,
        "programme": programme,
        "academic_risks": risks,
        "academic_risk": academic_risk,
        "behavior_risk": behavior_risk,
        "behavior_risk_max": behavior_risk_max,
        "behavior_risk_change": behavior_risk_change,
        "compliance_mean": compliance_mean,
        "anomaly_mean": anomaly_mean,
        "high_anomaly_weeks": high_anomaly_weeks,
        "compliance_trend": compliance_trend,
        "emotional_stress": emotional_stress,
        "evidence_coverage": coverage,
        "available_sources": 3,
        "expected_sources": 3,
        "trajectory": trajectory.state,
        "trajectory_explanation": trajectory.explanation,
        "priority_score": priority.score,
        "priority_level": priority.level,
        "base_priority_level": priority.level,
        "effective_priority": priority.level,
        "confidence": priority.confidence,
        "is_multi_modal_override": priority.is_multi_modal_override,
        "reason_codes": list(priority.reason_codes),
        "recommended_action": recommended_action,
        "owner_role": "Programme advisor" if priority.level == "P1" else "Student success coach",
        "due_date": str(date.today() + timedelta(days=3 if priority.level == "P1" else 7)),
        "status": "New",
        "follow_up": "Pending",
        "model_evidence": {
            "academic": f"Week 4 {risks[0]:.2f} -> Week 17 {risks[3]:.2f}",
            "behavioural": f"Mean weekly behavioural risk {behavior_risk:.2f} (anomaly {anomaly_mean:.2f})",
            "emotional": f"Stress signal {emotional_stress:.1f}/4.0",
        },
        "intervention_opportunity": opp_result,
        "data_mode": "live_stream",
    }
    return profile
