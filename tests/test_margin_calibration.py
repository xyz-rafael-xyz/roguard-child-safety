import math
import unittest

from eval.calibrate_margins import candidates, counts, predictions_for, select_thresholds
from roguard.review import APPLIES_TO, CATEGORIES


class MarginCalibrationTests(unittest.TestCase):
    def test_candidates_reject_nonfinite_and_out_of_range(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                candidates([value])
        self.assertEqual(candidates([0.2, 0.8]), (0.0, 0.5, 1.0))

    def test_joint_response_thresholds_preserve_both_categories(self):
        kinds = {"D1": "message", "R1": "routing_card", "A1": "response",
                 "P1": "permission_card", "G1": "gate_card", "S1": "response"}
        rows = []
        scores = {}
        for code in CATEGORIES:
            for positive in (False, True):
                identifier = f"{code}-{int(positive)}"
                kind = kinds[code]
                rows.append({"id": identifier, "source_kind": kind,
                             "labels": [code] if positive else []})
                scores[identifier] = {item: (0.8 if item == code and positive else 0.2)
                                      for item in CATEGORIES if kind in APPLIES_TO[item]}
        thresholds = select_thresholds(rows, scores)
        self.assertEqual(set(thresholds), set(CATEGORIES))
        self.assertTrue(all(math.isclose(value, 0.5) for value in thresholds.values()))
        self.assertEqual(counts(rows, predictions_for(rows, scores, thresholds))["exact_pairs"], 6)


if __name__ == "__main__":
    unittest.main()
