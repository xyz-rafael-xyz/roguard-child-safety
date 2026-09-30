import unittest
from collections import Counter
from pathlib import Path

from eval.verify_mmbert_v13 import verify
from roguard.review import ReviewError, load_approved, load_train_dev
from training.generate_advisory_holdout import build_rows

ROOT = Path(__file__).resolve().parents[1]


class V13ProtocolTests(unittest.TestCase):
    def test_fresh_advisory_pairs_reproduce_and_stay_out_of_fit(self):
        rows = load_approved(ROOT, ["batch-0025"])
        self.assertEqual(rows, build_rows())
        self.assertEqual(len(rows), 96)
        self.assertEqual(Counter(code for row in rows for code in row["labels"]),
                         {"D1": 24, "S1": 24})
        self.assertEqual(Counter(row["source_kind"] for row in rows),
                         {"message": 48, "response": 48})
        for negative, positive in zip(rows[::2], rows[1::2]):
            self.assertEqual(negative["labels"], [])
            self.assertEqual(len(positive["labels"]), 1)
            self.assertEqual(negative["source_kind"], positive["source_kind"])
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0021", "batch-0022", "batch-0025"], "ro")

    def test_frozen_advisory_metrics_recompute(self):
        self.assertTrue(verify(ROOT)["preregistered_success"])


if __name__ == "__main__":
    unittest.main()
