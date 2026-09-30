"""Prospective V27 split and rare-fact loss weighting invariants."""

from __future__ import annotations

import unittest

from roguard.v26_facts import truth_vector
from training.generate_v27_balanced import build_rows


class V27BalancedTests(unittest.TestCase):
    def test_held_out_styles_and_one_fact_pair_truth(self):
        seed = "fixture-only-v27-seed-20261001"
        rows = {split: build_rows(split, seed) for split in ("train", "dev", "test")}
        self.assertEqual({row["style_index"] for row in rows["train"]}, set(range(8)))
        self.assertEqual({row["style_index"] for row in rows["dev"]}, {8})
        self.assertEqual({row["style_index"] for row in rows["test"]}, {9})
        for split_rows in rows.values():
            for index in range(0, len(split_rows), 2):
                negative, positive = split_rows[index:index + 2]
                self.assertEqual(negative["labels"], [])
                self.assertEqual(len(positive["labels"]), 1)
                self.assertEqual(sum(negative["latent_facts"][key] !=
                                     positive["latent_facts"][key]
                                     for key in positive["latent_facts"]), 1)

    def test_rare_fact_values_have_equal_total_loss_weight(self):
        from training.v27_weights import fact_value_weights

        rows = build_rows("train", "fixture-only-v27-seed-20261001")
        positive_weights, negative_weights = fact_value_weights(rows)
        self.assertEqual(len(positive_weights), 6)
        self.assertGreater(negative_weights[0], positive_weights[0])
        for index in range(6):
            positives = negatives = 0
            for row in rows:
                truth, mask = truth_vector(row)
                if mask[index]:
                    positives += int(bool(truth[index]))
                    negatives += int(not truth[index])
            self.assertAlmostEqual(positives * positive_weights[index],
                                   negatives * negative_weights[index])


if __name__ == "__main__":
    unittest.main()
