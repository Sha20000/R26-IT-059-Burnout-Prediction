import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from meta_xai.opportunity_model import (
    FEATURE_COLUMNS,
    bootstrap_intervention_outcomes,
    predict,
    train,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "incoming" / "v1_oulad"
TRAINING_CSV = ROOT / "data" / "training" / "intervention_outcomes.csv"
MODEL_PATH = ROOT / "models" / "intervention_opportunity_model.joblib"


class OpportunityModelTests(unittest.TestCase):
    def test_target_bootstrapping_logic(self):
        df = bootstrap_intervention_outcomes(INPUT_DIR)
        self.assertEqual(len(df), 1000)
        self.assertIn("intervention_success", df.columns)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, df.columns)

        # Verify bootstrapping ground truth logic:
        # Success = (week17_risk < week8_risk) and (compliance_trend > 0)
        for _, row in df.iterrows():
            expected = 1 if (row["week17_risk"] < row["week8_risk"] and row["compliance_trend"] > 0) else 0
            self.assertEqual(int(row["intervention_success"]), expected)

    def test_predict_and_shap_explanations(self):
        sample = {
            "week4_risk": 0.35,
            "week8_risk": 0.40,
            "week12_risk": 0.38,
            "week17_risk": 0.30,
            "risk_change": -0.05,
            "recent_risk_change": -0.08,
            "behavior_risk": 0.25,
            "compliance_mean": 0.85,
            "anomaly_mean": 0.15,
            "high_anomaly_weeks": 0,
            "compliance_trend": 0.20,
            "emotional_stress": 3.5,
            "evidence_coverage": 1.0,
        }
        res = predict(MODEL_PATH, sample)

        self.assertIn("intervention_success_probability", res)
        self.assertIn("opportunity_band", res)
        self.assertIn("feature_contributions", res)
        self.assertIn("top_drivers", res)
        self.assertIn("why_explanation", res)

        # Check SHAP math
        contribs = res["feature_contributions"]
        self.assertEqual(len(contribs), len(FEATURE_COLUMNS))
        abs_pct_sum = sum(abs(c["percentage"]) for c in contribs)
        self.assertAlmostEqual(abs_pct_sum, 100.0, delta=1.5)

        # Verify why explanation string is generated
        self.assertTrue(len(res["why_explanation"]) > 0)


if __name__ == "__main__":
    unittest.main()
