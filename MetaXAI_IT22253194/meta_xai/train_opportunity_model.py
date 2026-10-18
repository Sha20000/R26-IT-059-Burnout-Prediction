from __future__ import annotations

import argparse
import json
from pathlib import Path

from .opportunity_model import bootstrap_intervention_outcomes, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Meta-XAI intervention-opportunity model")
    parser.add_argument("--bootstrap-from", type=Path, default=Path("data/incoming/v1_oulad"), help="Incoming directory to bootstrap labels from")
    parser.add_argument("--input", type=Path, default=Path("data/training/intervention_outcomes.csv"), help="Training data CSV path")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Output directory for model artifacts")
    args = parser.parse_args()

    if args.bootstrap_from and args.bootstrap_from.exists():
        print(f"Bootstrapping target labels from {args.bootstrap_from} -> {args.input}...")
        bootstrapped_df = bootstrap_intervention_outcomes(args.bootstrap_from, args.input)
        print(f"Bootstrapped {len(bootstrapped_df)} student records. Positive rate: {bootstrapped_df['intervention_success'].mean():.2%}")

    result = train(args.input, args.output_dir)
    print(json.dumps(result.__dict__, indent=2))
