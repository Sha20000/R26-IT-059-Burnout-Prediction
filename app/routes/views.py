from flask import Blueprint, current_app, render_template

from app.services.workspace import overview

views_bp = Blueprint("views", __name__)


@views_bp.get("/")
def dashboard():
    records = current_app.config["WORKSPACE_RECORDS"]
    return render_template("dashboard.html", records=records, summary=overview(records))
