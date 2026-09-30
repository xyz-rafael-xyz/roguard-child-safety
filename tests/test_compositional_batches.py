import unittest
from collections import Counter
from pathlib import Path

from roguard.review import CATEGORIES, ReviewError, load_approved, load_train_dev
from training.generate_compositional import SPECS, build_rows

ROOT = Path(__file__).resolve().parents[1]


class CompositionalBatchTests(unittest.TestCase):
    def test_each_split_reproduces_balanced_reference_pairs(self):
        for batch_id, (split, pairs_per_category) in SPECS.items():
            with self.subTest(batch=batch_id):
                rows = load_approved(ROOT, [batch_id])
                self.assertEqual(rows, build_rows(batch_id))
                self.assertEqual(len(rows), 12 * pairs_per_category)
                self.assertEqual({row["split"] for row in rows}, {split})
                self.assertEqual(Counter(code for row in rows for code in row["labels"]),
                                 {code: pairs_per_category for code in CATEGORIES})
                for negative, positive in zip(rows[::2], rows[1::2]):
                    self.assertEqual(negative["labels"], [])
                    self.assertEqual(positive["source_kind"], negative["source_kind"])
                    self.assertEqual(len(positive["labels"]), 1)
                    self.assertNotEqual(negative["text"], positive["text"])

    def test_test_split_cannot_enter_training_export(self):
        train, dev = load_train_dev(ROOT, ["batch-0014", "batch-0015"], "ro")
        self.assertEqual((len(train), len(dev)), (576, 96))
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0014", "batch-0015", "batch-0016"], "ro")


if __name__ == "__main__":
    unittest.main()
