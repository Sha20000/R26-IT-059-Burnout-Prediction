import json
import tempfile
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
        self.assertEqual(len(profiles), 2543)
        self.assertTrue(report["ready_for_fusion"])
        self.assertEqual(report["fully_matched_ids"], 2543)
        first = next(p for p in profiles if p["student_id"] == "CANONICAL_000001")
        self.assertEqual(first["student_id"], "CANONICAL_000001")
        self.assertEqual(first["academic_risks"], [0.5022, 0.1955, 0.3702, 0.2561])
        self.assertAlmostEqual(first["behavior_risk"], 0.2654, places=3)
        self.assertEqual(first["emotional_stress"], 1.3224)
        self.assertIn("priority_level", first)
        self.assertIn("intervention_success_probability", first["intervention_opportunity"])

    def test_batch_endpoint_writes_profiles(self):
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        test_db = Path(temp_dir.name) / "test_batch.db"
        app = create_app(db_path=test_db)
        output_path = ROOT / "data" / "processed" / "test_v1_evidence_profiles.csv"
        app.config["V1_OUTPUT_PATH"] = output_path
        try:
            response = app.test_client().post("/api/analyze/batch", json={})
            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            self.assertEqual(body["profile_count"], 2543)
            self.assertTrue(output_path.exists())
        finally:
            try:
                output_path.unlink(missing_ok=True)
                temp_dir.cleanup()
            except Exception:
                pass

    def test_capacity_endpoint_reorders_queue(self):
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        test_db = Path(temp_dir.name) / "test_capacity.db"
        app = create_app(db_path=test_db)
        client = app.test_client()
        res = client.post("/api/capacity", json={"limit": 5})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["advisor_capacity"], 5)
        self.assertEqual(data["p1_allocated"], 5)

        res_students = client.get("/api/students?effective_priority=P1")
        self.assertEqual(res_students.status_code, 200)
        p1s = res_students.get_json()
        self.assertEqual(len(p1s), 5)
        try:
            temp_dir.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
