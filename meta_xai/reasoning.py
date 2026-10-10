from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Trajectory:
    state: str
    severity: float
    urgency: float
    change: float
    explanation: str


@dataclass(frozen=True)
class InterventionPriority:
    score: float
    level: str
    confidence: str
    reason_codes: tuple[str, ...]
    is_multi_modal_override: bool = False


def classify_academic_trajectory(
    week4: float,
    week8: float,
    week12: float,
    week17: float,
) -> Trajectory:
    values = [max(0.0, min(1.0, value)) for value in (week4, week8, week12, week17)]
    change = values[-1] - values[0]
    recent_change = values[-1] - values[-2]
    severity = values[-1]
    urgency = max(0.0, min(1.0, (change + max(0.0, recent_change)) / 2))

    if severity >= 0.7 and change >= 0.25:
        state = "rapidly_escalating"
        explanation = "Academic risk increased substantially across the observation window."
    elif severity >= 0.7 and max(values) - min(values) <= 0.15:
        state = "persistent_high"
        explanation = "Academic risk remained consistently high."
    elif change <= -0.2:
        state = "improving"
        explanation = "Academic risk decreased across the observation window."
    elif severity < 0.4 and change < 0.15:
        state = "stable_low"
        explanation = "Academic risk remained comparatively low and stable."
    elif values[-1] >= 0.6 and values[0] < 0.5:
        state = "late_emerging"
        explanation = "Elevated academic risk emerged later in the observation window."
    else:
        state = "unstable"
        explanation = "Academic risk changed without a single dominant trajectory pattern."

    return Trajectory(state, round(severity, 4), round(urgency, 4), round(change, 4), explanation)


def calculate_priority(
    trajectory: Trajectory,
    actionability: float,
    evidence_confidence: float,
    opportunity_window: float,
    available_sources: int,
    expected_sources: int,
    emotional_stress: float = 0.0,
    high_anomaly_weeks: int | float = 0,
    behavior_risk: float = 0.0,
    intervention_success_probability: float | None = None,
) -> InterventionPriority:
    """
    Multi-Modal Priority Fusion:
    Calculates intervention priority from academic trajectory and contextual evidence.
    Applies deterministic overrides:
    - If emotional_stress >= 3.0 (Severe stress): Escalates directly to P1.
    - If high_anomaly_weeks >= 1 (Erratic/anomalous study pattern): Escalates directly to P1.
    These overrides trigger even if academic trajectory is stable_low.
    """
    factors = [trajectory.severity, trajectory.urgency, actionability, evidence_confidence, opportunity_window]
    score = sum(factors) / len(factors)
    reason_codes: list[str] = [f"ACADEMIC_{trajectory.state.upper()}"]

    if trajectory.urgency >= 0.6:
        reason_codes.append("RISK_CHANGE_URGENT")
    if available_sources < expected_sources:
        reason_codes.append("INCOMPLETE_EVIDENCE")

    # Base academic triage band
    if score >= 0.62 or trajectory.state in {"rapidly_escalating", "persistent_high"}:
        level = "P1"
    elif score >= 0.38 and trajectory.state != "stable_low":
        level = "P2"
    else:
        level = "P3"

    # Multi-Modal Deterministic Overrides
    is_override = False

    if emotional_stress >= 3.0:
        level = "P1"
        is_override = True
        score = max(score, 0.88)
        reason_codes.append("EMOTIONAL_STRESS_SEVERE")
    elif emotional_stress >= 2.0:
        reason_codes.append("EMOTIONAL_STRESS_ELEVATED")

    if high_anomaly_weeks >= 1:
        level = "P1"
        is_override = True
        score = max(score, 0.82)
        reason_codes.append("BEHAVIORAL_ANOMALY_TRIGGERED")
    elif behavior_risk >= 0.6:
        if level == "P3":
            level = "P2"
        reason_codes.append("BEHAVIORAL_RISK_ELEVATED")

    # Discordant catch annotation for viva demo pitch
    if is_override and trajectory.state in {"stable_low", "improving"}:
        reason_codes.append("MULTI_MODAL_DISCORDANT_CATCH")

    coverage = available_sources / expected_sources if expected_sources else 0.0
    confidence = (
        "High"
        if evidence_confidence >= 0.75 and coverage >= 0.8
        else "Medium"
        if evidence_confidence >= 0.45
        else "Limited"
    )

    return InterventionPriority(
        score=round(score, 4),
        level=level,
        confidence=confidence,
        reason_codes=tuple(reason_codes),
        is_multi_modal_override=is_override,
    )


def rank_triage_queue(
    student_pool: list[dict],
    advisor_capacity: int = 5,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Capacity-Aware Knapsack & Overflow Queue Manager:
    When more than `advisor_capacity` students qualify for P1, rank all P1 candidates
    by their XGBoost intervention success probability in descending order.
    The top K students remain in the active P1 queue; overflow candidates are
    shifted to P2 (Monitoring / Queued Review).
    """
    advisor_capacity = max(1, int(advisor_capacity))

    def _get_opportunity_prob(student: dict) -> float:
        opp = student.get("intervention_opportunity")
        if isinstance(opp, dict) and "intervention_success_probability" in opp:
            return float(opp["intervention_success_probability"])
        return float(student.get("priority_score", 0.0))

    p1_candidates: list[dict] = []
    other_students: list[dict] = []

    for item in student_pool:
        # Preserve original unconstrained priority as base_priority
        base_p = item.get("base_priority_level") or item.get("priority_level", "P3")
        item["base_priority_level"] = base_p
        if base_p == "P1":
            p1_candidates.append(item)
        else:
            other_students.append(item)

    # Sort P1 candidates mathematically by XGBoost success probability descending,
    # breaking ties with multi-modal priority score
    p1_candidates.sort(
        key=lambda s: (_get_opportunity_prob(s), float(s.get("priority_score", 0.0))),
        reverse=True,
    )

    # Allocate scarce P1 slots up to capacity limit
    allocated_p1: list[dict] = []
    overflow_p2: list[dict] = []

    for idx, student in enumerate(p1_candidates, start=1):
        if idx <= advisor_capacity:
            student["effective_priority"] = "P1"
            student["capacity_status"] = "Allocated P1 (High Impact)"
            student["capacity_rank"] = idx
            student["is_overflow"] = False
            allocated_p1.append(student)
        else:
            student["effective_priority"] = "P2"
            student["capacity_status"] = "Deferred to P2 (Capacity Limit Reached)"
            student["capacity_rank"] = idx
            student["is_overflow"] = True
            reasons = list(student.get("reason_codes", []))
            if "CAPACITY_OVERFLOW_TO_P2" not in reasons:
                reasons.append("CAPACITY_OVERFLOW_TO_P2")
            student["reason_codes"] = reasons
            overflow_p2.append(student)

    # Non-P1 students maintain their natural priority
    for student in other_students:
        base_p = student.get("base_priority_level", "P3")
        student["effective_priority"] = base_p
        student["capacity_status"] = "Standard Review"
        student["capacity_rank"] = None
        student["is_overflow"] = False

    # Combine back: Allocated P1 first, then Overflow P2, then Natural P2, then P3
    natural_p2 = [s for s in other_students if s.get("base_priority_level") == "P2"]
    natural_p3 = [s for s in other_students if s.get("base_priority_level") == "P3"]

    ordered_queue = allocated_p1 + overflow_p2 + natural_p2 + natural_p3

    summary = {
        "advisor_capacity": advisor_capacity,
        "total_students": len(student_pool),
        "p1_qualified": len(p1_candidates),
        "p1_allocated": len(allocated_p1),
        "p1_overflow_to_p2": len(overflow_p2),
        "effective_p1_count": len(allocated_p1),
        "effective_p2_count": len(overflow_p2) + len(natural_p2),
        "effective_p3_count": len(natural_p3),
    }

    return ordered_queue, summary
