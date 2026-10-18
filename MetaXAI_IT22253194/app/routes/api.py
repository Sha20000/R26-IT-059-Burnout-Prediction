import csv
import io
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file

from meta_xai.fusion import build_evidence_profiles, ingest_student_payload, write_profiles_csv
from meta_xai.ingestion import load_identity_mapping
from meta_xai.reasoning import rank_triage_queue
from app.services.analytics import build_analytics
from app.services.workspace import analyze_record, overview, update_case

api_bp = Blueprint("api", __name__)


def _get_records() -> list[dict]:
    return current_app.config["WORKSPACE_RECORDS"]


@api_bp.get("/students")
def list_students():
    records = _get_records()
    capacity = request.args.get("capacity")
    if capacity is not None and capacity.isdigit():
        records, _ = rank_triage_queue(records, advisor_capacity=int(capacity))
    priority = request.args.get("priority")
    effective_priority = request.args.get("effective_priority")
    status = request.args.get("status")
    query = request.args.get("q", "").lower().strip()

    if effective_priority in {"P1", "P2", "P3"}:
        records = [row for row in records if row.get("effective_priority", row.get("priority_level")) == effective_priority]
    elif priority in {"P1", "P2", "P3"}:
        records = [row for row in records if row.get("priority_level") == priority]

    if status:
        records = [row for row in records if row.get("status", "").lower() == status.lower()]
    if query:
        records = [
            row for row in records
            if query in row.get("student_id", "").lower() or query in row.get("student_name", "").lower()
        ]
    return jsonify(records)


@api_bp.get("/students/<student_id>")
def get_student(student_id: str):
    for record in _get_records():
        if record["student_id"] == student_id:
            return jsonify(record)
    return jsonify({"error": "Student not found"}), 404


@api_bp.route("/capacity", methods=["GET", "POST"])
def manage_capacity():
    records = _get_records()
    payload = request.get_json(silent=True) or {}
    capacity = request.args.get("limit") or payload.get("limit") or current_app.config.get("ADVISOR_CAPACITY", 5)
    try:
        capacity_int = max(1, int(capacity))
    except (TypeError, ValueError):
        capacity_int = 5

    ranked_records, summary = rank_triage_queue(records, advisor_capacity=capacity_int)
    current_app.config["WORKSPACE_RECORDS"] = ranked_records
    current_app.config["ADVISOR_CAPACITY"] = capacity_int
    return jsonify(summary)


@api_bp.post("/ingest")
def ingest_event():
    payload = request.get_json(silent=True) or {}
    if not payload:
        return jsonify({"error": "Empty payload received"}), 400

    input_dir = current_app.config.get("V1_INPUT_DIR")
    mapping_path = input_dir / "student_mapping.csv" if input_dir else None
    mapping_df = current_app.config.get("MAPPING_DF")
    if mapping_df is None and mapping_path and mapping_path.exists():
        mapping_df, _ = load_identity_mapping(mapping_path)
        current_app.config["MAPPING_DF"] = mapping_df

    model_path = current_app.config.get("OPPORTUNITY_MODEL_PATH")
    capacity = current_app.config.get("ADVISOR_CAPACITY", 5)

    items = payload if isinstance(payload, list) else [payload]
    new_profiles = []

    records = list(_get_records())
    existing_indices = {r["student_id"]: i for i, r in enumerate(records)}

    for item in items:
        profile = ingest_student_payload(item, mapping_df=mapping_df, model_path=model_path)
        new_profiles.append(profile)
        stu_id = profile["student_id"]
        if stu_id in existing_indices:
            records[existing_indices[stu_id]] = profile
        else:
            existing_indices[stu_id] = len(records)
            records.append(profile)

    ranked_records, summary = rank_triage_queue(records, advisor_capacity=capacity)
    current_app.config["WORKSPACE_RECORDS"] = ranked_records
    current_app.config["DATA_MODE"] = "LIVE STREAM"

    return jsonify({
        "status": "ingested",
        "ingested_count": len(new_profiles),
        "student": new_profiles[0] if len(new_profiles) == 1 else new_profiles,
        "queue_summary": summary,
        "total_records": len(ranked_records),
    })


@api_bp.post("/reset")
def reset_workspace():
    current_app.config["WORKSPACE_RECORDS"] = []
    current_app.config["DATA_MODE"] = "COLD START (LIVE READY)"
    return jsonify({
        "status": "reset",
        "total_records": 0,
        "message": "Workspace queue cleared for cold start live demonstration.",
    })


@api_bp.get("/overview")
def get_overview():
    return jsonify(overview(_get_records(), current_app.config["DATA_MODE"]))


@api_bp.get("/analytics")
def get_analytics():
    root = current_app.config["V1_INPUT_DIR"].parents[2]
    return jsonify(build_analytics(
        _get_records(),
        current_app.config["V1_INPUT_DIR"] / "manifest.json",
        root / "models" / "intervention_opportunity_metrics.json",
    ))


@api_bp.post("/analyze")
def analyze_student():
    try:
        return jsonify(analyze_record(request.get_json(silent=True) or {}))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@api_bp.post("/analyze/batch")
def analyze_batch():
    payload = request.get_json(silent=True) or {}
    input_dir = Path(payload.get("input_dir", current_app.config["V1_INPUT_DIR"]))
    mapping_path = Path(payload.get("mapping_path", input_dir / "student_mapping.csv"))
    output_path = Path(payload.get("output_path", current_app.config["V1_OUTPUT_PATH"]))
    try:
        profiles, alignment = build_evidence_profiles(
            input_dir,
            mapping_path,
            input_dir / "manifest.json",
            current_app.config.get("OPPORTUNITY_MODEL_PATH"),
        )
        write_profiles_csv(profiles, output_path)
        return jsonify({
            "profiles": profiles,
            "profile_count": len(profiles),
            "output_path": str(output_path),
            "alignment": alignment,
        })
    except (OSError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@api_bp.patch("/students/<student_id>/case")
def update_student_case(student_id: str):
    record = update_case(_get_records(), student_id, request.get_json(silent=True) or {})
    if record is None:
        return jsonify({"error": "Student not found"}), 404
    return jsonify(record)


@api_bp.get("/export/csv")
def export_csv():
    fieldnames = [
        "student_id", "student_name", "priority_level", "priority_score",
        "confidence", "trajectory", "recommended_action", "status", "follow_up",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for record in _get_records():
        writer.writerow({key: record.get(key) for key in fieldnames})
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="meta_xai_intervention_queue.csv",
    )
