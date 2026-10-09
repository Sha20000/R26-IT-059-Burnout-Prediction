import json
import unittest
from pathlib import Path

from app import create_app
from meta_xai.fusion import build_evidence_profiles

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "incoming" / "v1_oulad"
MODEL_PATH = ROOT / "models" / "intervention_opportunity_model.joblib"


class FusionTests(unittest.TestCase):
    def test_profiles_preserve_upstream_scores_and_alignment(self):
        profiles, report = build_evidence_profiles(
            INPUT_DIR,
            INPUT_DIR / "student_mapping.csv",
            INPUT_DIR / "manifest.json",
            MODEL_PATH,
        )
        self.assertEqual(len(profiles), 1000)
        self.assertTrue(report["ready_for_fusion"])
        self.assertEqual(report["fully_matched_ids"], 1000)
        first = profiles[0]
        self.assertEqual(first["student_id"], "CANONICAL_000001")
        self.assertEqual(first["academic_risks"], [0.35, 0.37, 0.41, 0.47])
        self.assertAlmostEqual(first["behavior_risk"], 0.0875)
        self.assertEqual(first["emotional_stress"], 4.0)
        self.assertIn("priority_level", first)
        self.assertIn("intervention_success_probability", first["intervention_opportunity"])

    def test_batch_endpoint_writes_profiles(self):
        app = create_app()
        output_path = ROOT / "data" / "processed" / "test_v1_evidence_profiles.csv"
        app.config["V1_OUTPUT_PATH"] = output_path
        try:
            response = app.test_client().post("/api/analyze/batch", json={})
            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            self.assertEqual(body["profile_count"], 1000)
            self.assertTrue(output_path.exists())
        finally:
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
