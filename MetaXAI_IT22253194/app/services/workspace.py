from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path

from meta_xai.reasoning import calculate_priority, classify_academic_trajectory
from meta_xai.opportunity_model import predict

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "intervention_opportunity_model.joblib"


PATTERNS = [
    ([0.18, 0.26, 0.51, 0.78], 0.73, 0.82, "Academic advisor check-in", "Programme advisor"),
    ([0.68, 0.65, 0.72, 0.76], 0.64, 0.66, "Academic advisor review", "Programme advisor"),
    ([0.72, 0.58, 0.42, 0.29], 0.31, 0.78, "Recognition and light-touch follow-up", "Student success coach"),
    ([0.22, 0.24, 0.28, 0.31], 0.19, 0.89, "Routine monitoring", "Student success coach"),
    ([0.32, 0.44, 0.61, 0.59], 0.58, 0.48, "Module support referral", "Module leader"),
    ([0.54, 0.38, 0.67, 0.71], 0.77, 0.42, "Human review of conflicting evidence", "Case coordinator"),
]

NAMES = [
    "Amina Rahman", "Daniel Perera", "Maya Fernando", "Lucas Silva", "Nethmi Jayasinghe",
    "Owen Martin", "Sofia Chen", "Kavindu Senanayake", "Priya Patel", "Adam Williams",
    "Tharushi De Alwis", "Elena Gomez", "Hassan Ali", "Isabelle Brown", "Ravi Kumar",
    "Noah Taylor", "Mihiri Gunawardena", "Ethan Davis",
]
PROGRAMMES = ["Computing", "Business Analytics", "Software Engineering", "Information Systems"]


def _record(index: int) -> dict:
    risks, behaviour, confidence, action, owner = PATTERNS[index % len(PATTERNS)]
    trajectory = classify_academic_trajectory(*risks)
    available_sources = 3 if index % 5 else 2
    emotional = 2 if index % 4 == 1 else 1 if index % 3 == 0 else 0
    priority = calculate_priority(
        trajectory,
        actionability=0.82 if action != "Human review of conflicting evidence" else 0.45,
        evidence_confidence=confidence,
        opportunity_window=0.85 if trajectory.urgency >= 0.5 else 0.58,
        available_sources=available_sources,
        expected_sources=3,
        emotional_stress=float(emotional),
        behavior_risk=float(behaviour),
    )
    student_id = f"DEMO-{index + 1:04d}"
    status = "Assigned" if index in (0, 1, 5) else "New"
    return {
        "student_id": student_id,
        "student_name": NAMES[index],
        "programme": PROGRAMMES[index % len(PROGRAMMES)],
        "cohort": "OULAD demo cohort / 2026-S1",
        "data_mode": "DEMO",
        "priority_score": priority.score,
        "priority_level": priority.level,
        "confidence": priority.confidence,
        "reason_codes": list(priority.reason_codes),
        "trajectory": trajectory.state,
        "trajectory_explanation": trajectory.explanation,
        "academic_risks": risks,
        "behaviour_risk": round(behaviour, 2),
        "emotional_stress": 2 if index % 4 == 1 else 1 if index % 3 == 0 else 0,
        "available_sources": available_sources,
        "expected_sources": 3,
        "evidence_coverage": round(available_sources / 3, 2),
        "recommended_action": action,
        "owner_role": owner,
        "due_date": str(date.today() + timedelta(days=3 if priority.level == "P1" else 7)),
        "status": status,
        "follow_up": "Pending" if status == "New" else "Scheduled",
        "model_evidence": {
            "academic": f"Week 4 {risks[0]:.2f} -> Week 17 {risks[3]:.2f}",
            "behavioural": f"Aggregated behavioural risk {behaviour:.2f}",
            "emotional": "Stress signal available" if index % 4 == 1 else "No elevated stress signal in demo fixture",
        },
    }


def build_demo_records() -> list[dict]:
    return [_record(index) for index in range(len(NAMES))]


def overview(records: list[dict], data_mode: str = "DEMO") -> dict:
    counts = {level: sum(row.get("effective_priority", row.get("priority_level")) == level for row in records) for level in ("P1", "P2", "P3")}
    cov = round(sum(row.get("evidence_coverage", 0.0) for row in records) / len(records), 2) if records else 1.0
    return {
        "total_students": len(records),
        "p1_cases": counts["P1"],
        "p2_cases": counts["P2"],
        "p3_cases": counts["P3"],
        "assigned_cases": sum(row.get("status") != "New" for row in records),
        "evidence_coverage": cov,
        "capacity": {"available_slots": 8, "weekly_case_limit": 12},
        "data_mode": data_mode,
    }


def _log_retraining_outcome(record: dict) -> None:
    log_path = Path(__file__).resolve().parents[2] / "data" / "training" / "intervention_outcomes_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if log_path.exists():
        try:
            entries = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    outcome_str = str(record.get("intervention_outcome", ""))
    is_success = 1 if "success" in outcome_str.lower() else 0
    entry = {
        "student_id": record.get("student_id"),
        "canonical_student_id": record.get("canonical_student_id"),
        "timestamp": str(date.today()),
        "intervention_outcome": outcome_str,
        "advisor_notes": record.get("advisor_notes", ""),
        "priority_level": record.get("priority_level"),
        "effective_priority": record.get("effective_priority"),
        "predicted_success_probability": (record.get("intervention_opportunity") or {}).get("intervention_success_probability"),
        "actual_label": is_success,
        "academic_risks": record.get("academic_risks"),
        "emotional_stress": record.get("emotional_stress"),
        "behavior_risk": record.get("behavior_risk"),
        "routed_to_retraining": True,
    }
    entries.append(entry)
    log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def update_case(records: list[dict], student_id: str, payload: dict) -> dict | None:
    for record in records:
        if record["student_id"] == student_id:
            if "status" in payload and payload["status"] in {"New", "Assigned", "Contacted", "Monitoring", "Resolved"}:
                record["status"] = payload["status"]
            if "follow_up" in payload and payload["follow_up"] in {"Pending", "Scheduled", "Complete"}:
                record["follow_up"] = payload["follow_up"]
            if "intervention_outcome" in payload:
                record["intervention_outcome"] = payload["intervention_outcome"]
                record["advisor_notes"] = payload.get("notes", "")
                record["resolved_at"] = str(date.today())
                record["status"] = "Resolved"
                _log_retraining_outcome(record)
            return deepcopy(record)
    return None


def analyze_record(payload: dict) -> dict:
    risks = payload.get("academic_risks", [])
    if len(risks) != 4:
        raise ValueError("academic_risks must contain exactly four values: Weeks 4, 8, 12, and 17")
    risks = [float(value) for value in risks]
    trajectory = classify_academic_trajectory(*risks)
    available_sources = int(payload.get("available_sources", 3))
    expected_sources = int(payload.get("expected_sources", 3))
    evidence_confidence = float(payload.get("evidence_confidence", 0.7))
    priority = calculate_priority(
        trajectory,
        float(payload.get("actionability", 0.7)),
        evidence_confidence,
        float(payload.get("opportunity_window", 0.7)),
        available_sources,
        expected_sources,
        emotional_stress=float(payload.get("emotional_stress", 0.0)),
        high_anomaly_weeks=float(payload.get("high_anomaly_weeks", 0.0)),
        behavior_risk=float(payload.get("behavior_risk", 0.0)),
    )
    result = {
        "trajectory": trajectory.state,
        "trajectory_explanation": trajectory.explanation,
        "priority_score": priority.score,
        "priority_level": priority.level,
        "base_priority_level": priority.level,
        "effective_priority": priority.level,
        "is_multi_modal_override": priority.is_multi_modal_override,
        "confidence": priority.confidence,
        "reason_codes": list(priority.reason_codes),
    }
    if MODEL_PATH.exists():
        result["opportunity_model"] = predict(MODEL_PATH, {
            "week4_risk": risks[0],
            "week8_risk": risks[1],
            "week12_risk": risks[2],
            "week17_risk": risks[3],
            "risk_change": trajectory.change,
            "recent_risk_change": risks[3] - risks[2],
            "behavior_risk": float(payload.get("behavior_risk", 0.0)),
            "compliance_mean": float(payload.get("compliance_mean", 0.8)),
            "anomaly_mean": float(payload.get("anomaly_mean", 0.2)),
            "high_anomaly_weeks": float(payload.get("high_anomaly_weeks", 0.0)),
            "compliance_trend": float(payload.get("compliance_trend", 0.0)),
            "emotional_stress": float(payload.get("emotional_stress", 0.0)),
            "evidence_coverage": available_sources / expected_sources if expected_sources else 1.0,
        })
    else:
        result["opportunity_model"] = {
            "status": "not_trained",
            "message": "Train the model with intervention outcomes before using opportunity predictions.",
        }
    return result
