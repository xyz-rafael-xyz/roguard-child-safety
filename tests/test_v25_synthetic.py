"""One-fact synthetic oracle and prospective reporting checks."""

from __future__ import annotations

import random
import unittest

from eval.v25_metrics import CELLS, evaluate, select_thresholds, target_passed
from training.generate_v25_synthetic import (_d1_oracle, _d1_pair, _s1_oracle,
                                             _s1_pair, build_rows)


class V25SyntheticTests(unittest.TestCase):
    def test_each_pair_flips_exactly_one_oracle_fact(self):
        for index in range(192):
            _, negative, positive = _d1_pair(index, random.Random(index))
            self.assertFalse(_d1_oracle(negative))
            self.assertTrue(_d1_oracle(positive))
            self.assertEqual(sum(vars(negative)[key] != vars(positive)[key]
                                 for key in vars(positive)), 1)
            _, negative, positive = _s1_pair(index)
            self.assertFalse(_s1_oracle(negative))
            self.assertTrue(_s1_oracle(positive))
            self.assertEqual(sum(vars(negative)[key] != vars(positive)[key]
                                 for key in vars(positive)), 1)

    def test_withheld_styles_have_balanced_cells_and_no_shared_text(self):
        rows = {split: build_rows(split, "fixture-seed-only-20261001")
                for split in ("train", "dev", "test")}
        self.assertEqual([len(rows[split]) for split in rows], [1536, 384, 384])
        self.assertFalse(set(row["text"] for row in rows["train"]) &
                         set(row["text"] for row in rows["dev"] + rows["test"]))
        self.assertFalse(set(row["text"] for row in rows["dev"]) &
                         set(row["text"] for row in rows["test"]))
        for split, split_rows in rows.items():
            for cell in CELLS:
                language, code = cell.split(":")
                negatives = [row for row in split_rows if row["language"] == language
                             and not row["labels"] and row["source_kind"] ==
                             ("message" if code == "D1" else "response")]
                positives = [row for row in split_rows if row["language"] == language
                             and row["labels"] == [code]]
                self.assertEqual(len(negatives), len(positives))
                self.assertEqual(len(positives), 192 if split == "train" else 48)

    def test_complete_pair_target_and_false_review_load(self):
        rows = build_rows("test", "fixture-seed-only-20261001")
        scores = [0.1 if not row["labels"] else 0.9 for row in rows]
        thresholds = select_thresholds(rows, scores)
        report = evaluate(rows, scores, thresholds)
        self.assertTrue(target_passed(report))
        self.assertEqual(report["exact_pairs"], 192)
        self.assertTrue(all(item["false_reviews"] == 0 for item in report["cells"].values()))
        scores[0] = 0.99
        damaged = evaluate(rows, scores, thresholds)
        self.assertEqual(damaged["cells"]["ro:D1"]["false_reviews"], 1)
        self.assertEqual(damaged["exact_pairs"], 191)
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            evaluate(rows, scores[:-1] + [float("nan")], thresholds)


if __name__ == "__main__":
    unittest.main()
