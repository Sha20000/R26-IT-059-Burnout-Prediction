"""
Meta-XAI Local Event Streamer Daemon (Phase 1)
Streams synthetic student records line-by-line to Flask /api/ingest to simulate live institutional activity.

Key features:
- Configurable streaming interval (--delay: 0.5s - 3s).
- Graceful termination on SIGINT/SIGTERM.
- Silent stdout with in-place ticker (prevents terminal buffer overflow and log bloat).
- Bounded file logging via RotatingFileHandler (max 1MB).
- Optional cold-start reset trigger (--reset-first).
"""
from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import sys
import time
import urllib.error
import urllib.request

# Ensure workspace root is in sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from meta_xai.ingestion import (
    load_academic,
    load_behaviour,
    load_emotional,
    load_identity_mapping,
)

RUNNING = True


def handle_shutdown(signum, frame):
    global RUNNING
    RUNNING = False
    print("\n[STREAMER] Shutdown signal received. Halting event stream cleanly...")


def setup_logger(log_file: Path | None) -> logging.Logger:
    logger = logging.getLogger("streamer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_file,
            maxBytes=1_000_000,  # 1 MB strict cap
            backupCount=1,
            encoding="utf-8",
        )
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


def trigger_reset(target_url: str, logger: logging.Logger) -> bool:
    reset_url = target_url.replace("/ingest", "/reset")
    try:
        req = urllib.request.Request(
            reset_url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print(f"[STREAMER] Workspace successfully reset to Cold Start via {reset_url}")
                logger.info("Workspace reset to Cold Start")
                return True
    except Exception as exc:
        print(f"[STREAMER] Warning: Reset request failed ({exc}). Continuing stream...")
        logger.warning(f"Reset request failed: {exc}")
    return False


def load_student_records(data_dir: Path) -> list[dict]:
    academic_table = load_academic(data_dir / "academic_predictions.csv")
    behaviour_table = load_behaviour(data_dir / "behavior_predictions.csv")
    emotional_table = load_emotional(data_dir / "emotional_predictions.csv")
    mapping_df, _ = load_identity_mapping(data_dir / "student_mapping.csv")

    if mapping_df is None:
        raise ValueError(f"student_mapping.csv not found in {data_dir}")

    acad_map = {str(r["student_id"]): r for r in academic_table.records}
    beh_map = {str(r["student_id"]): r for r in behaviour_table.records}
    emo_map = {str(r["student_id"]): r for r in emotional_table.records}

    records: list[dict] = []
    for _, row in mapping_df.iterrows():
        records.append({
            "canonical_student_id": str(row["canonical_student_id"]),
            "academic_student_id": str(row["academic_student_id"]),
            "behavior_student_id": str(row["behavior_student_id"]),
            "emotional_student_id": str(row["emotional_student_id"]),
            "cohort_id": str(row.get("cohort_id", "OULAD_COHORT_2026_01")),
            "academic": acad_map.get(str(row["academic_student_id"]), {}),
            "behaviour": beh_map.get(str(row["behavior_student_id"]), {}),
            "emotional": emo_map.get(str(row["emotional_student_id"]), {}),
        })

    return records


def post_event(target_url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        target_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return resp.status, body


def run_streamer(
    target_url: str,
    data_dir: Path,
    delay: float = 1.0,
    limit: int | None = None,
    loop: bool = False,
    reset_first: bool = False,
    log_file: Path | None = None,
):
    global RUNNING
    signal.signal(signal.SIGINT, handle_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_shutdown)

    logger = setup_logger(log_file)
    print("=" * 65)
    print("  META-XAI LOCAL EVENT STREAMER (PHASE 1)")
    print(f"  Target Endpoint : {target_url}")
    print(f"  Streaming Delay : {delay:.2f}s per event")
    print(f"  Data Source     : {data_dir}")
    print("=" * 65)

    if reset_first:
        trigger_reset(target_url, logger)

    records = load_student_records(data_dir)
    total_records = len(records) if limit is None else min(len(records), limit)
    records = records[:total_records]
    print(f"[STREAMER] Prepared {total_records} student event payloads.")
    print("[STREAMER] Streaming started. Press CTRL+C to halt gracefully.\n")

    iteration = 0
    total_sent = 0

    while RUNNING:
        iteration += 1
        for idx, student in enumerate(records, start=1):
            if not RUNNING:
                break

            try:
                status_code, body = post_event(target_url, student)
                total_sent += 1

                # Silent stdout: in-place status line updated every event
                pct = (idx / total_records) * 100.0
                stu_id = student["canonical_student_id"]
                queue_sum = body.get("queue_summary", {})
                p1_alloc = queue_sum.get("p1_allocated", "-")
                sys.stdout.write(
                    f"\r[STREAMER] Ingested: {idx}/{total_records} ({pct:5.1f}%) | "
                    f"Last: {stu_id} | HTTP {status_code} | P1 Queue: {p1_alloc} slots"
                )
                sys.stdout.flush()

                if idx % 50 == 0:
                    logger.info(f"Progress: {idx}/{total_records} ({pct:.1f}%) sent successfully")

            except urllib.error.URLError as err:
                sys.stdout.write(f"\n[STREAMER] Connection error: {err.reason}. Retrying in 3s...\n")
                sys.stdout.flush()
                logger.error(f"HTTP connection error: {err.reason}")
                time.sleep(3.0)
            except Exception as exc:
                sys.stdout.write(f"\n[STREAMER] Error sending {student['canonical_student_id']}: {exc}\n")
                sys.stdout.flush()
                logger.error(f"Error: {exc}")

            time.sleep(delay)

        if not loop or not RUNNING:
            break

        print(f"\n[STREAMER] Completed iteration {iteration}. Looping again...")
        logger.info(f"Completed iteration {iteration}")

    print(f"\n[STREAMER] Stream finished. Total events dispatched: {total_sent}.")
    logger.info(f"Streamer finished. Total events dispatched: {total_sent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-XAI Local Event Streamer Daemon")
    parser.add_argument("--target", type=str, default="http://127.0.0.1:5000/api/ingest", help="Target ingestion endpoint URL")
    parser.add_argument("--data-dir", type=Path, default=Path("data/incoming/v1_oulad"), help="Incoming synthetic data directory")
    parser.add_argument("--delay", type=float, default=1.0, help="Interval delay in seconds between events (default: 1.0)")
    parser.add_argument("--limit", type=int, default=None, help="Max records to stream (default: all)")
    parser.add_argument("--loop", action="store_true", help="Loop stream indefinitely")
    parser.add_argument("--reset-first", action="store_true", help="Reset workspace to empty Cold Start before streaming")
    parser.add_argument("--log-file", type=Path, default=Path("models/streamer.log"), help="Path for bounded rotating log file")

    args = parser.parse_args()
    run_streamer(
        target_url=args.target,
        data_dir=args.data_dir,
        delay=args.delay,
        limit=args.limit,
        loop=args.loop,
        reset_first=args.reset_first,
        log_file=args.log_file,
    )
