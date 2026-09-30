import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from eval.uk_contrast_suite import build_rows
from roguard.review import APPLIES_TO


ROOT = Path(__file__).resolve().parents[1]


class UkrainianContrastSuiteTests(unittest.TestCase):
    def test_draft_probe_reproduces_pairs_and_remains_out_of_training(self):
        path = ROOT / "data" / "synthetic" / "uk-contrast-0001.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(rows, build_rows())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["sha256"])
        self.assertEqual(manifest["taxonomy_status"], "draft_unreviewed")
        self.assertFalse(manifest["training_eligible"])
        self.assertEqual(Counter(row["labels"][0] for row in rows if row["labels"]),
                         Counter({code: 4 for code in ("D1", "R1", "A1", "P1", "G1", "S1")}))
        for left, right in zip(rows[::2], rows[1::2]):
            self.assertEqual(left["pair_id"], right["pair_id"])
            self.assertEqual(left["source_kind"], right["source_kind"])
            self.assertEqual(left["labels"], [])
            self.assertIn(right["source_kind"], APPLIES_TO[right["labels"][0]])


if __name__ == "__main__":
    unittest.main()
