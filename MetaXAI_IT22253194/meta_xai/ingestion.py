from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import AlignmentReport, SourceIssue, SourceTable

ACADEMIC_REQUIRED = {
    "student_id",
    "academic_risk",
    "week4_risk",
    "week8_risk",
    "week12_risk",
    "week17_risk",
}
BEHAVIOUR_REQUIRED = {"student_id", "behavioral_risk_score"}
EMOTIONAL_REQUIRED = {"student_id", "emotional_stress_score"}


def _read_csv(path: Path, source: str) -> tuple[pd.DataFrame | None, list[SourceIssue]]:
    if not path.exists():
        return None, [SourceIssue(source, "missing_file", f"File not found: {path}")]
    try:
        return pd.read_csv(path), []
    except Exception as exc:
        return None, [SourceIssue(source, "read_error", str(exc))]


def _check_columns(
    frame: pd.DataFrame | None,
    required: set[str],
    source: str,
    issues: list[SourceIssue],
) -> bool:
    if frame is None:
        return False
    missing = sorted(required - set(frame.columns))
    if missing:
        issues.append(SourceIssue(source, "missing_columns", "Required columns are missing", details={"columns": missing}))
        return False
    return True


def _check_ids(frame: pd.DataFrame, source: str, issues: list[SourceIssue]) -> set[str]:
    ids = frame["student_id"].dropna().astype(str).str.strip()
    if ids.empty:
        issues.append(SourceIssue(source, "empty_ids", "No non-empty student IDs were found"))
    if ids.duplicated().any():
        issues.append(
            SourceIssue(
                source,
                "duplicate_ids",
                "Duplicate student IDs require aggregation or a source-specific key",
                severity="warning",
                details={"duplicate_rows": int(ids.duplicated().sum())},
            )
        )
    return set(ids)


def _check_probability_range(frame: pd.DataFrame, columns: list[str], source: str, issues: list[SourceIssue]) -> None:
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        invalid = values[(values < 0) | (values > 1)]
        if not invalid.empty:
            issues.append(
                SourceIssue(
                    source,
                    "invalid_score_range",
                    f"{column} contains values outside [0, 1]",
                    details={"column": column, "invalid_count": int(len(invalid))},
                )
            )


def load_academic(path: Path) -> SourceTable:
    frame, issues = _read_csv(path, "academic")
    if not _check_columns(frame, ACADEMIC_REQUIRED, "academic", issues):
        return SourceTable("academic", [], set(), issues)
    assert frame is not None
    _check_probability_range(frame, sorted(ACADEMIC_REQUIRED - {"student_id"}), "academic", issues)
    frame = frame.drop_duplicates("student_id", keep="last").copy()
    records = frame.to_dict(orient="records")
    return SourceTable("academic", records, _check_ids(frame, "academic", issues), issues)


def load_behaviour(path: Path) -> SourceTable:
    frame, issues = _read_csv(path, "behaviour")
    if not _check_columns(frame, BEHAVIOUR_REQUIRED, "behaviour", issues):
        return SourceTable("behaviour", [], set(), issues)
    assert frame is not None
    _check_probability_range(frame, ["behavioral_risk_score", "curriculum_compliance", "anomaly_score"], "behaviour", issues)
    numeric_columns = [
        column
        for column in ["behavioral_risk_score", "curriculum_compliance", "anomaly_score", "high_anomaly_flag", "week"]
        if column in frame
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "week" in frame.columns:
        frame = frame.sort_values(["student_id", "week"])
    grouped = frame.groupby("student_id", dropna=True)
    aggregate = grouped.agg(
        behavior_risk_mean=("behavioral_risk_score", "mean"),
        behavior_risk_max=("behavioral_risk_score", "max"),
        compliance_mean=("curriculum_compliance", "mean"),
        anomaly_mean=("anomaly_score", "mean"),
        high_anomaly_weeks=("high_anomaly_flag", "sum"),
        latest_week=("week", "max"),
    ).reset_index()
    if "week" in frame:
        first = grouped["behavioral_risk_score"].first()
        last = grouped["behavioral_risk_score"].last()
        aggregate["behavior_risk_change"] = aggregate["student_id"].map(last - first)
        first_comp = grouped["curriculum_compliance"].first()
        last_comp = grouped["curriculum_compliance"].last()
        aggregate["compliance_trend"] = aggregate["student_id"].map(last_comp - first_comp)
    else:
        aggregate["compliance_trend"] = 0.0
    records = aggregate.to_dict(orient="records")
    return SourceTable("behaviour", records, _check_ids(aggregate, "behaviour", issues), issues)


def load_emotional(path: Path) -> SourceTable:
    frame, issues = _read_csv(path, "emotional")
    if not _check_columns(frame, EMOTIONAL_REQUIRED, "emotional", issues):
        return SourceTable("emotional", [], set(), issues)
    assert frame is not None
    frame["emotional_stress_score"] = pd.to_numeric(frame["emotional_stress_score"], errors="coerce")
    invalid = frame["emotional_stress_score"].dropna().loc[lambda values: (values < 0) | (values > 4)]
    if not invalid.empty:
        issues.append(SourceIssue("emotional", "invalid_stress_range", "Stress scores must be between 0 and 4"))
    frame = frame.drop_duplicates("student_id", keep="last").copy()
    records = frame.to_dict(orient="records")
    return SourceTable("emotional", records, _check_ids(frame, "emotional", issues), issues)


def load_identity_mapping(path: Path) -> tuple[pd.DataFrame | None, list[SourceIssue]]:
    frame, issues = _read_csv(path, "identity")
    if frame is None:
        return None, issues
    required = {"canonical_student_id", "academic_student_id", "behavior_student_id", "emotional_student_id"}
    if not _check_columns(frame, required, "identity", issues):
        return None, issues
    if frame["canonical_student_id"].duplicated().any():
        issues.append(SourceIssue("identity", "duplicate_canonical_ids", "Canonical IDs must be unique"))
    return frame.fillna(""), issues


def build_demo_mapping(sources: list[SourceTable]) -> pd.DataFrame:
    """Build an exact-ID mapping for a shared demo fixture only."""
    source_ids = {source.source: source.student_ids for source in sources}
    common_ids = set.intersection(*[ids for ids in source_ids.values() if ids])
    return pd.DataFrame(
        {
            "canonical_student_id": sorted(common_ids),
            "academic_student_id": sorted(common_ids),
            "behavior_student_id": sorted(common_ids),
            "emotional_student_id": sorted(common_ids),
        }
    )


def build_alignment_report(
    sources: list[SourceTable],
    mapping: pd.DataFrame | None,
    mapping_issues: list[SourceIssue] | None = None,
) -> AlignmentReport:
    source_counts = {source.source: len(source.records) for source in sources}
    source_unique_ids = {source.source: len(source.student_ids) for source in sources}
    issues = [issue for source in sources for issue in source.issues]
    issues.extend(mapping_issues or [])

    if mapping is None:
        non_empty = [source.student_ids for source in sources if source.student_ids]
        matched = len(set.intersection(*non_empty)) if non_empty else 0
        issues.append(SourceIssue("identity", "mapping_required", "Explicit identity mapping is required before cross-source fusion"))
        return AlignmentReport(source_counts, source_unique_ids, matched, 0, {}, issues)

    canonical_ids = set(mapping["canonical_student_id"].astype(str))
    coverage = {}
    for source, column in {
        "academic": "academic_student_id",
        "behaviour": "behavior_student_id",
        "emotional": "emotional_student_id",
    }.items():
        values = set(mapping[column].astype(str)) - {""}
        source_ids = next((item.student_ids for item in sources if item.source == source), set())
        matched = len(values & source_ids)
        coverage[source] = round(matched / len(canonical_ids), 4) if canonical_ids else 0.0

    source_sets = {
        source: next((item.student_ids for item in sources if item.source == source), set())
        for source in ["academic", "behaviour", "emotional"]
    }
    matched_ids = 0
    for _, row in mapping.iterrows():
        source_ids = {
            "academic": str(row["academic_student_id"]),
            "behaviour": str(row["behavior_student_id"]),
            "emotional": str(row["emotional_student_id"]),
        }
        if all(source_ids[source] and source_ids[source] in source_sets[source] for source in source_ids):
            matched_ids += 1
    return AlignmentReport(source_counts, source_unique_ids, matched_ids, len(canonical_ids), coverage, issues)


def report_as_dict(report: AlignmentReport) -> dict[str, Any]:
    return {
        "source_counts": report.source_counts,
        "source_unique_ids": report.source_unique_ids,
        "canonical_ids": report.canonical_ids,
        "fully_matched_ids": report.matched_ids,
        "coverage": report.coverage,
        "ready_for_fusion": report.is_ready_for_fusion,
        "issues": [
            {
                "source": issue.source,
                "code": issue.code,
                "message": issue.message,
                "severity": issue.severity,
                "details": issue.details,
            }
            for issue in report.issues
        ],
    }


def resolve_student_identity(
    student_id: str,
    mapping_df: pd.DataFrame,
) -> dict[str, str]:
    """
    Bridge OULAD_*, BEHAVIOR_*, and EMOTIONAL_* namespaces using student_mapping.csv.
    Matches across canonical_student_id, academic_student_id, behavior_student_id, or emotional_student_id.
    """
    student_id = str(student_id).strip()
    match = mapping_df[
        (mapping_df["canonical_student_id"] == student_id)
        | (mapping_df["academic_student_id"] == student_id)
        | (mapping_df["behavior_student_id"] == student_id)
        | (mapping_df["emotional_student_id"] == student_id)
    ]
    if match.empty:
        return {
            "canonical_student_id": student_id,
            "academic_student_id": student_id,
            "behavior_student_id": student_id,
            "emotional_student_id": student_id,
            "cohort_id": "UNKNOWN",
        }
    row = match.iloc[0]
    return {
        "canonical_student_id": str(row["canonical_student_id"]),
        "academic_student_id": str(row["academic_student_id"]),
        "behavior_student_id": str(row["behavior_student_id"]),
        "emotional_student_id": str(row["emotional_student_id"]),
        "cohort_id": str(row.get("cohort_id", "OULAD_COHORT_2026_01")),
    }
