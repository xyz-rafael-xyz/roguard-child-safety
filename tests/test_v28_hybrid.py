"""Frozen hybrid evidence routing and prospective split checks."""

from __future__ import annotations

import unittest

from roguard.v28_hybrid import abstract_body, compose, required_hypotheses
from training.generate_v28_hybrid import build_rows


class V28HybridTests(unittest.TestCase):
    def test_new_test_style_is_unseen_by_development(self):
        seed = "fixture-only-v28-seed-20261001"
        rows = {split: build_rows(split, seed) for split in ("train", "dev", "test")}
        self.assertEqual({row["style_index"] for row in rows["train"]}, set(range(9)))
        self.assertEqual({row["style_index"] for row in rows["dev"]}, {9})
        self.assertEqual({row["style_index"] for row in rows["test"]}, {10})
        self.assertFalse(set(row["text"] for row in rows["train"]) &
                         set(row["text"] for row in rows["dev"] + rows["test"]))

    def test_metadata_body_excludes_marker_and_closing(self):
        row = build_rows("dev", "fixture-only-v28-seed-20261001")[0]
        body = abstract_body(row)
        self.assertNotIn(row["text"].split(": ", 1)[0], body)
        self.assertNotIn(row["text"].rsplit(". ", 1)[1], body)
        with self.assertRaisesRegex(ValueError, "labeled body"):
            abstract_body({"language": "ro", "source_kind": "message", "text": "unstructured"})

    def test_fact_sources_are_fixed_per_language_and_category(self):
        encoder = [0.0] * 6
        ro_d1 = {"language": "ro", "source_kind": "message"}
        uk_d1 = {"language": "uk", "source_kind": "message"}
        s1 = {"language": "uk", "source_kind": "response"}
        self.assertEqual(required_hypotheses(ro_d1), (0, 1, 2, 3))
        self.assertEqual(required_hypotheses(uk_d1), (0,))
        self.assertEqual(required_hypotheses(s1), (4,))
        self.assertEqual(compose(ro_d1, encoder, {0: 1., 1: 1., 2: 1., 3: 0.})[0], 1.)
        self.assertEqual(compose(uk_d1, [0., 1., 0., 1., 0., 0.], {0: 1.})[0], 1.)
        self.assertEqual(compose(s1, [0., 0., 0., 0., 0., 0.], {4: 1.})[0], 1.)
        with self.assertRaisesRegex(ValueError, "required fact evidence"):
            compose(ro_d1, encoder, {0: 1.})


if __name__ == "__main__":
    unittest.main()
