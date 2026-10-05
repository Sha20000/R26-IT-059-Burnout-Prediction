from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingestion import (
    build_alignment_report,
    build_demo_mapping,
    load_academic,
    load_behaviour,
    load_emotional,
    load_identity_mapping,
    report_as_dict,
)


def build_report(input_dir: Path, mapping_path: Path, demo: bool = False) -> dict:
    sources = [
        load_academic(input_dir / "academic_predictions.csv"),
        load_behaviour(input_dir / "behavior_predictions.csv"),
        load_emotional(input_dir / "emotional_predictions.csv"),
    ]
    mapping, mapping_issues = load_identity_mapping(mapping_path)
    if demo and mapping is None:
        mapping = build_demo_mapping(sources)
        mapping_issues = []
    return report_as_dict(build_alignment_report(sources, mapping, mapping_issues))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Meta-XAI source outputs before fusion")
    parser.add_argument("--input-dir", type=Path, default=Path("data/incoming"))
    parser.add_argument("--mapping", type=Path, default=Path("data/identity/student_mapping.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/alignment_report.json"))
    parser.add_argument("--demo", action="store_true", help="Allow exact shared IDs to form a demo-only mapping")
    args = parser.parse_args()

    report = build_report(args.input_dir, args.mapping, demo=args.demo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
