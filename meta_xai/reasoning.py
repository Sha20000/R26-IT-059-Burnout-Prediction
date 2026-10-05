from __future__ import annotations

from dataclasses import dataclass


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
) -> InterventionPriority:
    factors = [trajectory.severity, trajectory.urgency, actionability, evidence_confidence, opportunity_window]
    score = sum(factors) / len(factors)
    reason_codes = [f"ACADEMIC_{trajectory.state.upper()}"]
    if trajectory.urgency >= 0.6:
        reason_codes.append("RISK_CHANGE_URGENT")
    if available_sources < expected_sources:
        reason_codes.append("INCOMPLETE_EVIDENCE")

    if score >= 0.62 or trajectory.state in {"rapidly_escalating", "persistent_high"}:
        level = "P1"
    elif score >= 0.38:
        level = "P2"
    else:
        level = "P3"

    coverage = available_sources / expected_sources if expected_sources else 0.0
    confidence = "High" if evidence_confidence >= 0.75 and coverage >= 0.8 else "Medium" if evidence_confidence >= 0.45 else "Limited"
    return InterventionPriority(round(score, 4), level, confidence, tuple(reason_codes))
