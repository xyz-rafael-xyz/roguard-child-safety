"""Prospective calibration selection and batch provenance invariants."""

import runpy
import unittest
from pathlib import Path

from roguard.review import load_approved

ROOT = Path(__file__).resolve().parents[1]
STUDY = runpy.run_path(str(ROOT / "eval/run_mmbert_v17.py"))


class V17ProtocolTests(unittest.TestCase):
    def test_new_batches_are_attested_and_disjoint(self):
        dev = load_approved(ROOT, ["batch-0031"])
        test = load_approved(ROOT, ["batch-0032"])
        self.assertEqual((len(dev), len(test)), (96, 96))
        self.assertEqual({row["split"] for row in dev}, {"dev"})
        self.assertEqual({row["split"] for row in test}, {"test"})
        self.assertFalse({row["text"] for row in dev} & {row["text"] for row in test})
        for rows in (dev, test):
            self.assertEqual(sum(row["labels"] == ["D1"] for row in rows), 24)
            self.assertEqual(sum(row["labels"] == ["S1"] for row in rows), 24)

    def test_selector_prefers_specific_cutoff_when_recall_is_retained(self):
        rows = []
        scores = []
        for index in range(24):
            for positive in (False, True):
                row_id = f"toy-{index}-{positive}"
                expected = ["D1"] if positive else []
                rows.append({"id": row_id, "language": "ro", "source_kind": "message", "labels": expected})
                value = 0.99 if positive else 0.6 if index < 8 else 0.1
                scores.append({"id": row_id, "expected": expected, "strict_parse": True,
                               "scores": {"D1": value}})
        cutoff, report, eligible, _ = STUDY["select_d1"](rows, scores)
        self.assertTrue(eligible)
        self.assertGreater(cutoff, 0.6)
        self.assertLessEqual(cutoff, 0.99)
        self.assertEqual(report["exact_pairs"], 24)
        self.assertEqual(report["false_review_on_negatives"], 0)


if __name__ == "__main__":
    unittest.main()
