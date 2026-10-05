import logging

from flask import Flask

from app.routes.api import api_bp
from app.routes.views import views_bp
from app.services.workspace import build_demo_records


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["WORKSPACE_RECORDS"] = build_demo_records()
    app.config["DATA_MODE"] = "DEMO"
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(views_bp)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return app
