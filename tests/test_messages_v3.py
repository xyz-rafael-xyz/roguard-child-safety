"""Fictional-message generator and dependency-free inference contracts."""

import gzip
import json
import math
import unittest

from roguard.message_model import SyntheticMessageModel, assess_message
from training.generate_messages_v3 import generate, validate_splits


class MessageV3Tests(unittest.TestCase):
    def test_fictional_generator_has_balanced_disjoint_pairs(self):
        summary = validate_splits()
        self.assertEqual({key: summary[key]["rows"] for key in summary},
                         {"train": 5120, "dev": 384, "test": 384})
        for split in ("train", "dev", "test"):
            rows = generate(split)
            self.assertEqual(sum(row["label"] for row in rows), len(rows) // 2)
            pair_ids = {row["pair_id"] for row in rows}
            self.assertEqual(len(pair_ids), len(rows) // 2)
            self.assertEqual({row["language"] for row in rows}, {"ro", "uk"})

    def test_model_scores_without_echoing_text_or_authorizing_action(self):
        data = gzip.compress(json.dumps({
            "format": "roguard-character-tfidf-v1", "language": "ro",
            "ngram_range": [2, 2], "cutoff": 0.5, "intercept": 0.0,
            "features": [["ab", 1.0, 2.0]]}).encode(), mtime=0)
        model = SyntheticMessageModel(data)
        self.assertAlmostEqual(model.score("ab"), 1 / (1 + math.exp(-2)))
        self.assertEqual(model.score("xy"), 0.5)
        result = assess_message({"language": "ro", "declared_source_role": "minor",
                                 "text": "ab"}, model)
        self.assertEqual(result["advisory_code"], "REVIEW_SUPPORT_SIGNAL")
        self.assertNotIn("text", result)
        self.assertFalse(result["message_echoed"])
        self.assertFalse(result["real_world_accuracy_established"])
        self.assertFalse(result["external_action_taken"])
        with self.assertRaises(ValueError):
            assess_message({"language": "uk", "declared_source_role": "minor",
                            "text": "ab"}, model)
        with self.assertRaises(ValueError):
            assess_message({"language": "ro", "declared_source_role": "adult",
                            "text": "ab"}, model)


if __name__ == "__main__":
    unittest.main()
