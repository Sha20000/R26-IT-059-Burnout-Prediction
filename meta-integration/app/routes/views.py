from pathlib import Path

from flask import Blueprint, current_app, render_template

from app.services import explain, ingestion, integration

views_bp = Blueprint("views", __name__)


def _base_dir() -> Path:
    return Path(current_app.root_path).parent


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


@views_bp.get("/")
def dashboard():
    records = _build_records()

    total = len(records)
    counts = {
        "Low": len([r for r in records if r.get("final_risk") == "Low"]),
        "Medium": len([r for r in records if r.get("final_risk") == "Medium"]),
        "High": len([r for r in records if r.get("final_risk") == "High"]),
    }

    scores = [r["final_score"] for r in records if isinstance(r.get("final_score"), (int, float))]
    avg_score = round(sum(scores) / len(scores), 3) if scores else None

    return render_template(
        "dashboard.html",
        records=records,
        total=total,
        counts=counts,
        avg_score=avg_score,
    )
