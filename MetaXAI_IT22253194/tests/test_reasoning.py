import unittest
from meta_xai.reasoning import (
    Trajectory,
    calculate_priority,
    classify_academic_trajectory,
    rank_triage_queue,
)


class ReasoningTests(unittest.TestCase):
    def test_multi_modal_override_severe_emotional_stress(self):
        # Academic risk is low and flat (stable_low)
        trajectory = classify_academic_trajectory(0.10, 0.12, 0.14, 0.15)
        self.assertEqual(trajectory.state, "stable_low")

        # Without severe stress, priority is P3
        base_priority = calculate_priority(
            trajectory,
            actionability=0.7,
            evidence_confidence=0.8,
            opportunity_window=0.7,
            available_sources=3,
            expected_sources=3,
            emotional_stress=1.0,
            high_anomaly_weeks=0,
        )
        self.assertEqual(base_priority.level, "P3")
        self.assertFalse(base_priority.is_multi_modal_override)

        # With severe stress (3.5/4.0), deterministic override triggers P1
        override_priority = calculate_priority(
            trajectory,
            actionability=0.7,
            evidence_confidence=0.8,
            opportunity_window=0.7,
            available_sources=3,
            expected_sources=3,
            emotional_stress=3.5,
            high_anomaly_weeks=0,
        )
        self.assertEqual(override_priority.level, "P1")
        self.assertTrue(override_priority.is_multi_modal_override)
        self.assertIn("EMOTIONAL_STRESS_SEVERE", override_priority.reason_codes)
        self.assertIn("MULTI_MODAL_DISCORDANT_CATCH", override_priority.reason_codes)

    def test_multi_modal_override_behavioral_anomaly(self):
        # Academic trajectory is stable_low
        trajectory = classify_academic_trajectory(0.15, 0.16, 0.17, 0.18)
        self.assertEqual(trajectory.state, "stable_low")

        # With high anomaly flag triggered (erratic logins)
        anomaly_priority = calculate_priority(
            trajectory,
            actionability=0.7,
            evidence_confidence=0.8,
            opportunity_window=0.7,
            available_sources=3,
            expected_sources=3,
            emotional_stress=1.0,
            high_anomaly_weeks=1,
        )
        self.assertEqual(anomaly_priority.level, "P1")
        self.assertTrue(anomaly_priority.is_multi_modal_override)
        self.assertIn("BEHAVIORAL_ANOMALY_TRIGGERED", anomaly_priority.reason_codes)
        self.assertIn("MULTI_MODAL_DISCORDANT_CATCH", anomaly_priority.reason_codes)

    def test_capacity_constrained_knapsack_ranking(self):
        # Create a synthetic pool of 8 P1 students with different XGBoost probabilities
        pool = []
        probs = [0.95, 0.88, 0.82, 0.75, 0.68, 0.55, 0.42, 0.31]
        for idx, p in enumerate(probs):
            pool.append({
                "student_id": f"P1_STU_{idx+1}",
                "priority_level": "P1",
                "priority_score": 0.85,
                "intervention_opportunity": {
                    "intervention_success_probability": p,
                },
            })

        # Add 3 P2 and 2 P3 students
        for idx in range(3):
            pool.append({
                "student_id": f"P2_STU_{idx+1}",
                "priority_level": "P2",
                "priority_score": 0.50,
            })
        for idx in range(2):
            pool.append({
                "student_id": f"P3_STU_{idx+1}",
                "priority_level": "P3",
                "priority_score": 0.25,
            })

        # Test with capacity limit of 5
        ranked, summary = rank_triage_queue(pool, advisor_capacity=5)

        self.assertEqual(summary["advisor_capacity"], 5)
        self.assertEqual(summary["p1_qualified"], 8)
        self.assertEqual(summary["p1_allocated"], 5)
        self.assertEqual(summary["p1_overflow_to_p2"], 3)
        self.assertEqual(summary["effective_p1_count"], 5)
        self.assertEqual(summary["effective_p2_count"], 6)  # 3 original + 3 overflow

        # Top 5 must be the highest probability students
        allocated_ids = [s["student_id"] for s in ranked if s["effective_priority"] == "P1"]
        self.assertEqual(allocated_ids, ["P1_STU_1", "P1_STU_2", "P1_STU_3", "P1_STU_4", "P1_STU_5"])

        # Remaining 3 P1 students overflowed to P2
        overflow_ids = [s["student_id"] for s in ranked if s.get("is_overflow")]
        self.assertEqual(overflow_ids, ["P1_STU_6", "P1_STU_7", "P1_STU_8"])
        for s in ranked:
            if s["student_id"] in overflow_ids:
                self.assertEqual(s["effective_priority"], "P2")
                self.assertIn("CAPACITY_OVERFLOW_TO_P2", s["reason_codes"])

        # Dynamically lower capacity to 2 (simulating UI slider drag)
        re_ranked, re_summary = rank_triage_queue(pool, advisor_capacity=2)
        self.assertEqual(re_summary["p1_allocated"], 2)
        self.assertEqual(re_summary["p1_overflow_to_p2"], 6)
        re_allocated_ids = [s["student_id"] for s in re_ranked if s["effective_priority"] == "P1"]
        self.assertEqual(re_allocated_ids, ["P1_STU_1", "P1_STU_2"])


if __name__ == "__main__":
    unittest.main()
