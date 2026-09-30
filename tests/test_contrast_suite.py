import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from eval.contrast_suite import build_rows
from roguard.review import APPLIES_TO


ROOT = Path(__file__).resolve().parents[1]


class ContrastSuiteTests(unittest.TestCase):
    def test_frozen_pairs_flip_one_expected_category(self):
        path = ROOT / "data" / "synthetic" / "contrast-0001.jsonl"
        stored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(stored, build_rows())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["sha256"])
        self.assertFalse(manifest["training_eligible"])
        self.assertEqual(len(stored), 48)
        self.assertEqual(Counter(row["labels"][0] for row in stored if row["labels"]),
                         Counter({code: 4 for code in ("D1", "R1", "A1", "P1", "G1", "S1")}))
        for left, right in zip(stored[::2], stored[1::2]):
            self.assertEqual((left["pair_id"], left["source_kind"]),
                             (right["pair_id"], right["source_kind"]))
            self.assertEqual((left["variant"], right["variant"]), (0, 1))
            self.assertEqual(left["labels"], [])
            self.assertEqual(len(right["labels"]), 1)
            self.assertIn(right["source_kind"], APPLIES_TO[right["labels"][0]])
            self.assertNotEqual(left["text"], right["text"])


if __name__ == "__main__":
    unittest.main()
