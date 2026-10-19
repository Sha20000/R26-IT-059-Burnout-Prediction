import os
from pathlib import Path
import tempfile
import unittest

from app.db import (
    clear_students_in_db,
    fetch_all_students,
    fetch_intervention_logs,
    get_db_stats,
    init_db,
    save_or_update_students,
    update_case_in_db,
)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "test_burnout.db"
        init_db(self.db_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_init_and_stats(self):
        stats = get_db_stats(self.db_path)
        self.assertTrue(stats["exists"])
        self.assertEqual(stats["total_students"], 0)
        self.assertEqual(stats["total_intervention_audit_logs"], 0)

    def test_save_and_fetch_students(self):
        records = [
            {
                "student_id": "TEST_001",
                "canonical_student_id": "10001",
                "student_name": "Test Student",
                "programme": "Computing",
                "priority_level": "P1",
                "priority_score": 0.88,
                "confidence": "High",
                "status": "New",
                "academic_risks": [0.2, 0.4, 0.6, 0.8],
                "behavior_risk": 0.75,
                "emotional_stress": 2,
            }
        ]
        save_or_update_students(records, self.db_path)
        students = fetch_all_students(self.db_path)
        self.assertEqual(len(students), 1)
        self.assertEqual(students[0]["student_id"], "TEST_001")
        self.assertEqual(students[0]["priority_level"], "P1")
        self.assertEqual(students[0]["academic_risks"], [0.2, 0.4, 0.6, 0.8])

    def test_case_resolution_and_audit_log(self):
        records = [
            {
                "student_id": "TEST_002",
                "canonical_student_id": "10002",
                "student_name": "Resolution Student",
                "priority_level": "P1",
                "priority_score": 0.92,
                "confidence": "High",
                "status": "New",
                "intervention_opportunity": {"intervention_success_probability": 0.84},
            }
        ]
        save_or_update_students(records, self.db_path)

        # Resolve the case
        payload = {
            "status": "Resolved",
            "intervention_outcome": "Student Responded & Improved",
            "notes": "Held pastoral session. Reduced study workload.",
        }
        updated = update_case_in_db("TEST_002", payload, self.db_path)
        self.assertIsNotNone(updated)
        self.assertEqual(updated["status"], "Resolved")
        self.assertEqual(updated["intervention_outcome"], "Student Responded & Improved")

        # Verify relational audit table
        logs = fetch_intervention_logs(self.db_path)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["student_id"], "TEST_002")
        self.assertEqual(logs[0]["actual_label"], 1)
        self.assertEqual(logs[0]["advisor_notes"], "Held pastoral session. Reduced study workload.")

    def test_clear_students(self):
        records = [{"student_id": "TEST_003", "priority_score": 0.5}]
        save_or_update_students(records, self.db_path)
        self.assertEqual(len(fetch_all_students(self.db_path)), 1)
        clear_students_in_db(self.db_path)
        self.assertEqual(len(fetch_all_students(self.db_path)), 0)


if __name__ == "__main__":
    unittest.main()
