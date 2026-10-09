from flask import Blueprint, current_app, render_template

from app.services.analytics import build_analytics
from app.services.workspace import overview

views_bp = Blueprint("views", __name__)


@views_bp.get("/")
def dashboard():
    records = current_app.config["WORKSPACE_RECORDS"]
    return render_template(
        "dashboard.html",
        records=records,
        summary=overview(records, current_app.config["DATA_MODE"]),
        data_mode=current_app.config["DATA_MODE"],
    )


@views_bp.get("/analytics")
def analytics():
    root = current_app.config["V1_INPUT_DIR"].parents[2]
    payload = build_analytics(
        current_app.config["WORKSPACE_RECORDS"],
        current_app.config["V1_INPUT_DIR"] / "manifest.json",
        root / "models" / "intervention_opportunity_metrics.json",
    )
    return render_template("analytics.html", analytics=payload)
