import unittest
from collections import Counter
from pathlib import Path

from roguard.review import CATEGORIES, ReviewError, load_approved, load_train_dev
from training.generate_encoder_holdout import build_rows

ROOT = Path(__file__).resolve().parents[1]


class EncoderHoldoutTests(unittest.TestCase):
    def test_sealed_pairs_reproduce_and_cannot_enter_training(self):
        rows = load_approved(ROOT, ["batch-0019"])
        self.assertEqual(rows, build_rows())
        self.assertEqual(len(rows), 144)
        self.assertEqual(Counter(code for row in rows for code in row["labels"]),
                         {code: 12 for code in CATEGORIES})
        for negative, positive in zip(rows[::2], rows[1::2]):
            self.assertEqual(negative["labels"], [])
            self.assertEqual(len(positive["labels"]), 1)
            self.assertEqual(negative["source_kind"], positive["source_kind"])
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0014", "batch-0015", "batch-0019"], "ro")


if __name__ == "__main__":
    unittest.main()
