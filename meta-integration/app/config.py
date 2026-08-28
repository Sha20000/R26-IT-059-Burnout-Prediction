import json
import os
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "weights.json"


def get_config_path() -> Path:
    override = os.getenv("APP_CONFIG_PATH")
    if override:
        return Path(override)
    return DEFAULT_CONFIG_PATH


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = ["weights", "thresholds", "override", "sources"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config missing keys: {missing}")

    return config
