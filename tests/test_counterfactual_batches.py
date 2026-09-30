import unittest
from collections import Counter
from pathlib import Path

from roguard.review import ReviewError, load_approved, load_train_dev, reproduce_registered
from training.generate_counterfactual import SPECS, build_rows
from training.generate_binary_challenge import build_rows as build_binary_challenge
from training.generate_margin_challenge import build_rows as build_margin_challenge


ROOT = Path(__file__).resolve().parents[1]


class CounterfactualBatchTests(unittest.TestCase):
    def test_historical_batch_still_reproduces_after_gate_reporting_changes(self):
        rows = load_approved(ROOT, ["batch-0008"])
        self.assertEqual(rows, reproduce_registered(ROOT, "batch-0008"))

    def test_registered_batches_reproduce_and_keep_splits_separate(self):
        for batch_id, (split, pair_count, repeats) in SPECS.items():
            with self.subTest(batch=batch_id):
                rows = load_approved(ROOT, [batch_id])
                self.assertEqual(rows, build_rows(batch_id))
                self.assertEqual(len(rows), 12 * pair_count * repeats)
                self.assertEqual({row["split"] for row in rows}, {split})
                labels = Counter(code for row in rows for code in row["labels"])
                self.assertEqual(labels, {code: pair_count * repeats for code in ("D1", "R1", "A1", "P1", "G1", "S1")})
                for negative, positive in zip(rows[::2], rows[1::2]):
                    self.assertEqual(negative["source_kind"], positive["source_kind"])
                    self.assertEqual(negative["labels"], [])
                    self.assertEqual(len(positive["labels"]), 1)
                    self.assertNotEqual(negative["text"], positive["text"])

    def test_future_test_cannot_enter_trainer(self):
        train, dev = load_train_dev(ROOT, ["batch-0009", "batch-0010"], "ro")
        self.assertEqual((len(train), len(dev)), (288, 36))
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0009", "batch-0010", "batch-0011"], "ro")
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0009", "batch-0010", "batch-0012"], "ro")
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0009", "batch-0010", "batch-0013"], "ro")

    def test_binary_challenge_is_a_separate_frozen_test(self):
        rows = load_approved(ROOT, ["batch-0012"])
        self.assertEqual(rows, build_binary_challenge())
        self.assertEqual(len(rows), 48)
        self.assertEqual(Counter(code for row in rows for code in row["labels"]),
                         {code: 4 for code in ("D1", "R1", "A1", "P1", "G1", "S1")})
        for negative, positive in zip(rows[::2], rows[1::2]):
            self.assertEqual(negative["labels"], [])
            self.assertEqual(positive["source_kind"], negative["source_kind"])
            self.assertNotEqual(positive["text"], negative["text"])

    def test_margin_challenge_is_a_separate_frozen_test(self):
        rows = load_approved(ROOT, ["batch-0013"])
        self.assertEqual(rows, build_margin_challenge())
        self.assertEqual(len(rows), 96)
        self.assertEqual(Counter(code for row in rows for code in row["labels"]),
                         {code: 8 for code in ("D1", "R1", "A1", "P1", "G1", "S1")})
        for negative, positive in zip(rows[::2], rows[1::2]):
            self.assertEqual(negative["labels"], [])
            self.assertEqual(positive["source_kind"], negative["source_kind"])
            self.assertNotEqual(positive["text"], negative["text"])


if __name__ == "__main__":
    unittest.main()
