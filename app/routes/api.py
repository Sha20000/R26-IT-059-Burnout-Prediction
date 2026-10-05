import csv
import io

from flask import Blueprint, current_app, jsonify, request, send_file

from app.services.workspace import analyze_record, overview, update_case

api_bp = Blueprint("api", __name__)


def _get_records() -> list[dict]:
    return current_app.config["WORKSPACE_RECORDS"]


@api_bp.get("/students")
def list_students():
    records = _get_records()
    priority = request.args.get("priority")
    status = request.args.get("status")
    query = request.args.get("q", "").lower().strip()
    if priority in {"P1", "P2", "P3"}:
        records = [row for row in records if row["priority_level"] == priority]
    if status:
        records = [row for row in records if row["status"].lower() == status.lower()]
    if query:
        records = [row for row in records if query in row["student_id"].lower() or query in row["student_name"].lower()]
    return jsonify(records)


@api_bp.get("/students/<student_id>")
def get_student(student_id: str):
    for record in _get_records():
        if record["student_id"] == student_id:
            return jsonify(record)
    return jsonify({"error": "Student not found"}), 404


@api_bp.get("/overview")
def get_overview():
    return jsonify(overview(_get_records()))


@api_bp.post("/analyze")
def analyze_student():
    try:
        return jsonify(analyze_record(request.get_json(silent=True) or {}))
    except (TypeError, ValueError) as exc:
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
