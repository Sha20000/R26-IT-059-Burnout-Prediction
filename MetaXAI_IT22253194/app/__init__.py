import logging
from pathlib import Path

from flask import Flask

from app.routes.api import api_bp
from app.routes.views import views_bp
from app.services.workspace import build_demo_records
from meta_xai.fusion import build_evidence_profiles


def create_app() -> Flask:
    app = Flask(__name__)
    root = Path(__file__).resolve().parents[1]
    app.config["WORKSPACE_RECORDS"] = build_demo_records()
    app.config["DATA_MODE"] = "DEMO"
    app.config["V1_INPUT_DIR"] = root / "data" / "incoming" / "v1_oulad"
    app.config["V1_OUTPUT_PATH"] = root / "data" / "processed" / "v1_evidence_profiles.csv"
    app.config["OPPORTUNITY_MODEL_PATH"] = root / "models" / "intervention_opportunity_model.joblib"
    try:
        records, _ = build_evidence_profiles(
            app.config["V1_INPUT_DIR"],
            app.config["V1_INPUT_DIR"] / "student_mapping.csv",
            app.config["V1_INPUT_DIR"] / "manifest.json",
            app.config["OPPORTUNITY_MODEL_PATH"],
        )
        app.config["WORKSPACE_RECORDS"] = records
        app.config["DATA_MODE"] = "SYNTHETIC DEMO"
    except (OSError, TypeError, ValueError) as exc:
        logging.warning("V1 data unavailable; using demo workspace: %s", exc)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(views_bp)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return app
