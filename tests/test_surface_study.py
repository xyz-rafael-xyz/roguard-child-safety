import unittest
from collections import Counter
from pathlib import Path

from roguard.review import CATEGORIES, ReviewError, load_approved, load_train_dev
from training.generate_surface_holdout import build_rows as build_test
from training.generate_surface_train import build_rows as build_fit

ROOT = Path(__file__).resolve().parents[1]


class SurfaceStudyTests(unittest.TestCase):
    def test_new_splits_reproduce_and_test_is_excluded(self):
        for batch, build, count, positive_per_code in (
            ("batch-0021", build_fit, 2304, 192),
            ("batch-0022", build_fit, 144, 12),
            ("batch-0023", build_test, 144, 12),
        ):
            with self.subTest(batch=batch):
                rows = load_approved(ROOT, [batch])
                self.assertEqual(rows, build(batch))
                self.assertEqual(len(rows), count)
                self.assertEqual(Counter(code for row in rows for code in row["labels"]),
                                 {code: positive_per_code for code in CATEGORIES})
                for negative, positive in zip(rows[::2], rows[1::2]):
                    self.assertEqual(negative["labels"], [])
                    self.assertEqual(negative["source_kind"], positive["source_kind"])
                    self.assertEqual(len(positive["labels"]), 1)
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0021", "batch-0022", "batch-0023"], "ro")


if __name__ == "__main__":
    unittest.main()
