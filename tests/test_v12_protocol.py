import math
import unittest
from collections import Counter
from pathlib import Path

from eval.select_mmbert_v12 import (apply_thresholds, select_thresholds,
                                    validated_inputs)
from eval.verify_mmbert_v12 import verify
from roguard.pair_eval import evaluate_pairs
from roguard.review import CATEGORIES, ReviewError, load_approved, load_train_dev
from training.generate_surface_holdout_v12 import build_rows

ROOT = Path(__file__).resolve().parents[1]


class V12ProtocolTests(unittest.TestCase):
    def test_new_holdout_reproduces_and_cannot_enter_fit(self):
        rows = load_approved(ROOT, ["batch-0024"])
        self.assertEqual(rows, build_rows())
        self.assertEqual(len(rows), 144)
        self.assertEqual(Counter(code for row in rows for code in row["labels"]),
                         {code: 12 for code in CATEGORIES})
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0021", "batch-0022", "batch-0024"], "ro")

    def test_development_choice_is_reproducible_without_test_labels(self):
        rows, items, _ = validated_inputs(ROOT)
        thresholds, report, counts = select_thresholds(rows, items)
        self.assertEqual(set(thresholds), set(CATEGORIES))
        self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1
                            for value in thresholds.values()))
        self.assertEqual(report, evaluate_pairs(rows, apply_thresholds(items, thresholds)))
        self.assertEqual(set(counts), {"D1", "R1", "P1", "G1", "A1+S1"})

    def test_frozen_test_decisions_recompute_from_saved_scores(self):
        self.assertFalse(verify(ROOT)["preregistered_success"])


if __name__ == "__main__":
    unittest.main()
