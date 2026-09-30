"""Latent-fact composition and prospective wording partition checks."""

from __future__ import annotations

import unittest

from eval.v26_metrics import evaluate, target_passed
from roguard.v26_facts import decision_score, make_input, truth_vector
from training.generate_v26_facts import build_rows


class V26FactTests(unittest.TestCase):
    def test_latent_facts_compose_exactly_to_typed_labels(self):
        for split in ("train", "dev", "test"):
            rows = build_rows(split, "fixture-only-v26-seed-20261001")
            self.assertEqual(len(rows), 1536 if split == "train" else 384)
            for row in rows:
                truth, mask = truth_vector(row)
                self.assertEqual(sum(mask), 4 if row["source_kind"] == "message" else 2)
                self.assertEqual(decision_score(truth, row["source_kind"]),
                                 float(bool(row["labels"])))
                self.assertNotIn("latent_facts", make_input(row))
                without_truth = {key: value for key, value in row.items()
                                 if key != "latent_facts"}
                self.assertEqual(make_input(row), make_input(without_truth))

    def test_development_and_test_styles_are_unseen_during_fit(self):
        seed = "fixture-only-v26-seed-20261001"
        rows = {split: build_rows(split, seed) for split in ("train", "dev", "test")}
        self.assertEqual({row["style_index"] for row in rows["train"]}, set(range(7)))
        self.assertEqual({row["style_index"] for row in rows["dev"]}, {7})
        self.assertEqual({row["style_index"] for row in rows["test"]}, {8, 9})
        self.assertFalse(set(row["text"] for row in rows["train"]) &
                         set(row["text"] for row in rows["dev"] + rows["test"]))

    def test_target_uses_explicit_new_style_and_changed_factor(self):
        rows = build_rows("test", "fixture-only-v26-seed-20261001")
        scores = [0.0 if not row["labels"] else 1.0 for row in rows]
        cutoffs = {cell: 0.5 for cell in ("ro:D1", "ro:S1", "uk:D1", "uk:S1")}
        report = evaluate(rows, scores, cutoffs)
        self.assertTrue(target_passed(report))
        for item in report["cells"].values():
            self.assertEqual(set(item["by_wording_style"]), {"8", "9"})
            self.assertEqual(sum(group["pairs"] for group in
                                 item["by_wording_style"].values()), 48)


if __name__ == "__main__":
    unittest.main()
