"""V29 fixed evidence routing and prospective wording partition."""

from __future__ import annotations

import unittest

from roguard.v29_hybrid import compose, required_hypotheses
from training.generate_v29_hybrid import build_rows


class V29HybridTests(unittest.TestCase):
    def test_unseen_test_wording_and_one_fact_pairs(self):
        seed = "fixture-only-v29-seed-20261001"
        rows = {split: build_rows(split, seed) for split in ("train", "dev", "test")}
        self.assertEqual({row["style_index"] for row in rows["train"]}, set(range(10)))
        self.assertEqual({row["style_index"] for row in rows["dev"]}, {10})
        self.assertEqual({row["style_index"] for row in rows["test"]}, {11})
        for split_rows in rows.values():
            for index in range(0, len(split_rows), 2):
                negative, positive = split_rows[index:index + 2]
                self.assertEqual((negative["labels"], positive["labels"]),
                                 ([], ["D1" if negative["source_kind"] == "message" else "S1"]))
                self.assertEqual(sum(negative["latent_facts"][key] !=
                                     positive["latent_facts"][key]
                                     for key in negative["latent_facts"]), 1)

    def test_romanian_anchor_comes_from_encoder(self):
        row = {"language": "ro", "source_kind": "message"}
        self.assertEqual(required_hypotheses(row), (0, 2, 3))
        encoder = [0., 0.25, 0., 0., 0., 0.]
        score, facts = compose(row, encoder, {0: 1., 2: 1., 3: 0.})
        self.assertEqual(score, 0.25)
        self.assertEqual(facts[1], 0.25)
        with self.assertRaisesRegex(ValueError, "required fact evidence"):
            compose(row, encoder, {0: 1., 1: 1., 2: 1., 3: 0.})


if __name__ == "__main__":
    unittest.main()
