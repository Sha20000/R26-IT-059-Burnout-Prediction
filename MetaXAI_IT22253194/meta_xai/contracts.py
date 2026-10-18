from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceIssue:
    source: str
    code: str
    message: str
    severity: str = "error"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceTable:
    source: str
    records: list[dict[str, Any]]
    student_ids: set[str]
    issues: list[SourceIssue] = field(default_factory=list)


@dataclass
class AlignmentReport:
    source_counts: dict[str, int]
    source_unique_ids: dict[str, int]
    matched_ids: int
    canonical_ids: int
    coverage: dict[str, float]
    issues: list[SourceIssue] = field(default_factory=list)

    @property
    def is_ready_for_fusion(self) -> bool:
        return self.canonical_ids > 0 and not any(
            issue.severity == "error" for issue in self.issues
        )
