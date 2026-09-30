import hashlib
import unittest

from eval.compare_v4 import summarize
from roguard.prompt_v4 import make_prompt_v4


class BinaryComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "d0", "language": "ro", "source_kind": "message", "text": "simbol zero", "labels": []},
            {"id": "d1", "language": "ro", "source_kind": "message", "text": "simbol unu", "labels": ["D1"]},
            {"id": "r0", "language": "ro", "source_kind": "routing_card", "text": "regulă zero", "labels": []},
            {"id": "r1", "language": "ro", "source_kind": "routing_card", "text": "regulă unu", "labels": ["R1"]},
        ]

    def test_pair_metric_detects_both_sides_flagged(self):
        output = {"predictions": [
            {"id": row["id"], "expected": row["labels"], "predicted": predicted, "strict_parse": True}
            for row, predicted in zip(self.rows, (["D1"], ["D1"], [], ["R1"]))
        ]}
        report = summarize(self.rows, output, "ro-v4-adapter")
        self.assertEqual((report["exact_rows"], report["exact_pairs"],
                          report["correct_direction_flips"], report["false_review_on_negatives"]),
                         (3, 1, 1, 1))

    def test_prompt_hash_and_raw_decision_are_rechecked(self):
        row = self.rows[0]
        digest = hashlib.sha256(make_prompt_v4(row, "D1").encode("utf-8")).hexdigest()
        item = {"id": "d0", "expected": [], "predicted": [], "strict_parse": True,
                "raw": {"D1": "nu"}, "prompt_sha256": {"D1": digest}}
        output = {"outputs": [item]}
        self.assertEqual(summarize([row], output, "romistral-base")["exact_rows"], 1)
        item["prompt_sha256"]["D1"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "provenance"):
            summarize([row], output, "romistral-base")


if __name__ == "__main__":
    unittest.main()
