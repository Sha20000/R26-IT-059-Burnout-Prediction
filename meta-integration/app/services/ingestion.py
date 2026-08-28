from functools import reduce
from pathlib import Path

import pandas as pd


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / raw_path


def _load_source(source: dict, base_dir: Path) -> pd.DataFrame:
    file_path = _resolve_path(base_dir, source["file"])
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    df = pd.read_csv(file_path)
    id_col = source.get("id_col", "student_id")
    score_col = source["score_col"]
    score_key = source["score_key"]
    reason_col = source.get("reason_col")

    columns = [id_col, score_col]
    if reason_col:
        columns.append(reason_col)

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path.name}: {missing}")

    df = df[columns].copy()
    df = df.rename(columns={id_col: "student_id", score_col: score_key})

    if reason_col:
        df = df.rename(columns={reason_col: f"{score_key}_reason"})

    return df


def load_records(config: dict, base_dir: Path) -> list:
    dataframes = [_load_source(source, base_dir) for source in config["sources"]]
    merged = reduce(lambda left, right: left.merge(right, on="student_id", how="outer"), dataframes)

    merged = merged.sort_values("student_id")
    merged = merged.where(pd.notnull(merged), None)

    return merged.to_dict(orient="records")
