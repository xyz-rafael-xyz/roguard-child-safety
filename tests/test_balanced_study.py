import tempfile
import unittest
from collections import Counter
from pathlib import Path

from roguard.review import CATEGORIES, ReviewError, load_approved, load_train_dev
from training.generate_balanced_holdout import build_rows
from training.prepare_balanced_mlx import prepare_balanced
from training.prepare_mlx import prepare

ROOT = Path(__file__).resolve().parents[1]


class BalancedStudyTests(unittest.TestCase):
    def test_new_holdout_is_attested_and_excluded_from_training(self):
        rows = load_approved(ROOT, ["batch-0018"])
        self.assertEqual(rows, build_rows())
        self.assertEqual(len(rows), 144)
        self.assertEqual(Counter(code for row in rows for code in row["labels"]),
                         {code: 12 for code in CATEGORIES})
        for negative, positive in zip(rows[::2], rows[1::2]):
            self.assertEqual(negative["labels"], [])
            self.assertEqual(negative["source_kind"], positive["source_kind"])
            self.assertEqual(len(positive["labels"]), 1)
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0014", "batch-0015", "batch-0018"], "ro")

    def test_only_training_exposure_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            original = prepare(ROOT, ["batch-0014", "batch-0015"], "ro", directory / "original", "v4")
            balanced = prepare_balanced(ROOT, directory / "balanced")
            self.assertEqual((original["train_tasks"], balanced["train_tasks"]), (960, 1728))
            self.assertEqual(balanced["dev_tasks"], 128)
            self.assertEqual(original["valid_sha256"], balanced["valid_sha256"])
            self.assertEqual(original["taxonomy_sha256"], balanced["taxonomy_sha256"])
            self.assertEqual(original["train_rows"], balanced["train_rows"])
            for code in CATEGORIES:
                self.assertEqual(balanced["train_tasks_per_category"][code], {"da": 144, "nu": 144})


if __name__ == "__main__":
    unittest.main()
