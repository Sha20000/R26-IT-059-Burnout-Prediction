import logging
from pathlib import Path

from flask import Flask

from app.db import fetch_all_students, init_db, save_or_update_students
from app.routes.api import api_bp
from app.routes.views import views_bp
from app.services.workspace import build_demo_records
from meta_xai.fusion import build_evidence_profiles


def create_app(db_path: Path | str | None = None) -> Flask:
    app = Flask(__name__)
    root = Path(__file__).resolve().parents[1]
    db_file = Path(db_path) if db_path else (root / "data" / "burnout_guard.db")
    app.config["DB_PATH"] = db_file
    init_db(db_file)

    app.config["V1_INPUT_DIR"] = root / "data" / "incoming" / "v1_oulad"
    app.config["V1_OUTPUT_PATH"] = root / "data" / "processed" / "v1_evidence_profiles.csv"
    app.config["OPPORTUNITY_MODEL_PATH"] = root / "models" / "intervention_opportunity_model.joblib"
    app.config["DATA_MODE"] = "SQLITE DATABASE (OULAD)"

    # Seed or hydrate from SQLite database
    existing_students = fetch_all_students(db_file)
    if existing_students:
        app.config["WORKSPACE_RECORDS"] = existing_students
        app.config["DATA_MODE"] = "SQLITE DATABASE (OULAD)"
    else:
        try:
            records, _ = build_evidence_profiles(
                app.config["V1_INPUT_DIR"],
                app.config["V1_INPUT_DIR"] / "student_mapping.csv",
                app.config["V1_INPUT_DIR"] / "manifest.json",
                app.config["OPPORTUNITY_MODEL_PATH"],
            )
            save_or_update_students(records, db_file)
            app.config["WORKSPACE_RECORDS"] = records
            app.config["DATA_MODE"] = "SQLITE DATABASE (OULAD)"
        except (OSError, TypeError, ValueError) as exc:
            logging.warning("V1 data unavailable; using demo workspace: %s", exc)
            demo_records = build_demo_records()
            save_or_update_students(demo_records, db_file)
            app.config["WORKSPACE_RECORDS"] = demo_records
            app.config["DATA_MODE"] = "SQLITE DEMO"

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(views_bp)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return app
