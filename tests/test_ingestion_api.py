import unittest
from pathlib import Path

from app import create_app
from meta_xai.ingestion import resolve_student_identity, load_identity_mapping

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "incoming" / "v1_oulad"


class IngestionApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_identity_resolution_across_namespaces(self):
        mapping_df, _ = load_identity_mapping(INPUT_DIR / "student_mapping.csv")
        self.assertIsNotNone(mapping_df)

        # Lookup by academic ID
        res_by_acad = resolve_student_identity("OULAD_000001", mapping_df)
        self.assertEqual(res_by_acad["canonical_student_id"], "CANONICAL_000001")
        self.assertEqual(res_by_acad["behavior_student_id"], "BEHAVIOR_000001")
        self.assertEqual(res_by_acad["emotional_student_id"], "EMOTIONAL_000001")

        # Lookup by behavior ID
        res_by_beh = resolve_student_identity("BEHAVIOR_000042", mapping_df)
        self.assertEqual(res_by_beh["canonical_student_id"], "CANONICAL_000042")
        self.assertEqual(res_by_beh["academic_student_id"], "OULAD_000042")

        # Lookup by emotional ID
        res_by_emo = resolve_student_identity("EMOTIONAL_000100", mapping_df)
        self.assertEqual(res_by_emo["canonical_student_id"], "CANONICAL_000100")

    def test_reset_workspace_cold_start(self):
        # Reset workspace
        res = self.client.post("/api/reset")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["total_records"], 0)

        # Verify overview handles empty workspace without error
        overview_res = self.client.get("/api/overview")
        self.assertEqual(overview_res.status_code, 200)
        overview_data = overview_res.get_json()
        self.assertEqual(overview_data["total_students"], 0)
        self.assertEqual(overview_data["p1_cases"], 0)

    def test_ingest_single_student_and_discordant_override(self):
        self.client.post("/api/reset")

        # Ingest a student with stable_low academic grades but severe emotional stress (4.0)
        discordant_payload = {
            "canonical_student_id": "CANONICAL_TEST_001",
            "academic": {
                "student_id": "OULAD_TEST_001",
                "week4_risk": 0.12,
                "week8_risk": 0.14,
                "week12_risk": 0.11,
                "week17_risk": 0.10,
                "academic_risk": 0.10,
                "programme": "Computing",
            },
            "behaviour": {
                "student_id": "BEHAVIOR_TEST_001",
                "behavior_risk_mean": 0.20,
                "compliance_mean": 0.90,
                "anomaly_mean": 0.15,
                "high_anomaly_weeks": 0,
                "compliance_trend": 0.10,
            },
            "emotional": {
                "student_id": "EMOTIONAL_TEST_001",
                "emotional_stress_score": 4.0,  # Severe stress
            },
        }

        res = self.client.post("/api/ingest", json=discordant_payload)
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["status"], "ingested")
        self.assertEqual(body["total_records"], 1)

        student = body["student"]
        self.assertEqual(student["student_id"], "CANONICAL_TEST_001")
        self.assertEqual(student["trajectory"], "stable_low")

        # Multi-modal override should have caught them and escalated to P1!
        self.assertEqual(student["effective_priority"], "P1")
        self.assertTrue(student["is_multi_modal_override"])
        self.assertIn("EMOTIONAL_STRESS_SEVERE", student["reason_codes"])
        self.assertIn("MULTI_MODAL_DISCORDANT_CATCH", student["reason_codes"])

        # Check TreeSHAP explanations attached
        self.assertIn("intervention_opportunity", student)
        opp = student["intervention_opportunity"]
        self.assertIn("intervention_success_probability", opp)
        self.assertIn("feature_contributions", opp)
        self.assertIn("why_explanation", opp)

    def test_batch_ingest_and_capacity_sorting(self):
        self.client.post("/api/reset")

        batch = []
        for i in range(1, 9):
            batch.append({
                "canonical_student_id": f"CANONICAL_BATCH_{i:03d}",
                "academic": {
                    "week4_risk": 0.70,
                    "week8_risk": 0.75,
                    "week12_risk": 0.80,
                    "week17_risk": 0.85,
                    "academic_risk": 0.85,
                },
                "behaviour": {
                    "behavior_risk_mean": 0.60,
                    "compliance_mean": 0.50,
                    "anomaly_mean": 0.40,
                    "high_anomaly_weeks": 1,
                    "compliance_trend": 0.10 if i <= 4 else -0.20,
                },
                "emotional": {
                    "emotional_stress_score": 3.0,
                },
            })

        res = self.client.post("/api/ingest", json=batch)
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["total_records"], 8)

        # Capacity is 5: top 5 should be allocated P1, remaining 3 should be overflow to P2
        summary = body["queue_summary"]
        self.assertEqual(summary["advisor_capacity"], 5)
        self.assertEqual(summary["p1_allocated"], 5)
        self.assertEqual(summary["p1_overflow_to_p2"], 3)


if __name__ == "__main__":
    unittest.main()
