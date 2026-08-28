import csv
import io
import time
from pathlib import Path

from flask import Blueprint, current_app, jsonify, send_file

from app.services import explain, ingestion, integration

api_bp = Blueprint("api", __name__)


def _base_dir() -> Path:
    return Path(current_app.root_path).parent


def _get_cache() -> dict:
    return current_app.config.setdefault("DATA_CACHE", {"timestamp": 0, "records": []})


def _build_records() -> list:
    app_config = current_app.config["APP_CONFIG"]
    base_dir = _base_dir()

    raw_records = ingestion.load_records(app_config, base_dir)
    enriched = []
    for record in raw_records:
        meta = integration.compute_final(record, app_config)
        explanation = explain.build_explanation(record, meta)
        merged = {**record, **meta, **explanation}
        enriched.append(merged)

    return enriched


def _get_records() -> list:
    cache = _get_cache()
    app_config = current_app.config["APP_CONFIG"]
    refresh_seconds = app_config.get("refresh_seconds", 30)

    if time.time() - cache["timestamp"] > refresh_seconds:
        cache["records"] = _build_records()
        cache["timestamp"] = time.time()

    return cache["records"]


@api_bp.get("/students")
def list_students():
    return jsonify(_get_records())


@api_bp.get("/students/<student_id>")
def get_student(student_id: str):
    for record in _get_records():
        if record.get("student_id") == student_id:
            return jsonify(record)
    return jsonify({"error": "Student not found"}), 404


@api_bp.get("/export/csv")
def export_csv():
    records = _get_records()
    output = io.StringIO()

    fieldnames = [
        "student_id",
        "academic_score",
        "emotion_score",
        "behavior_score",
        "engagement_score",
        "final_score",
        "final_risk",
        "override_applied",
        "summary",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow({key: record.get(key) for key in fieldnames})

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="student_risk_export.csv",
    )
