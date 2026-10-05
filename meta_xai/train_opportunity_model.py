from __future__ import annotations

import argparse
import json
from pathlib import Path

from .opportunity_model import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Meta-XAI intervention-opportunity model")
    parser.add_argument("--input", type=Path, default=Path("data/training/intervention_outcomes.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    result = train(args.input, args.output_dir)
    print(json.dumps(result.__dict__, indent=2))
