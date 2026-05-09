import logging
from flask import Flask

from app.config import get_config_path, load_config
from app.routes.api import api_bp
from app.routes.views import views_bp


def create_app() -> Flask:
    app = Flask(__name__)

    config_path = get_config_path()
    app_config = load_config(config_path)
    app.config["APP_CONFIG"] = app_config

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(views_bp)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    return app
