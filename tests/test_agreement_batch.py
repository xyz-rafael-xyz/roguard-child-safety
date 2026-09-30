"""Batch reviewer agreement reports decisions without card values."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from roguard import compare_evidence_batches


def d1(anchor, language="ro"):
    return {"language": language, "disclosure": {
        "source_role": "minor", "safety_or_support_anchor": anchor,
        "indirect_or_repeated_support_pattern": True,
        "explicit_support_request": False,
    }}


def record(item_id, card):
    return {"item_id": item_id, "card": card}


class AgreementBatchTests(unittest.TestCase):
    def test_complete_disagreement_matrix_and_unknown_exclusion(self):
        left = [record(f"item-{index}", d1(value)) for index, value in
                enumerate((False, True, False, True, None), 1)]
        right = [record(f"item-{index}", d1(value)) for index, value in
                 enumerate((False, True, True, False, None), 1)]
        report = compare_evidence_batches(left, right)
        self.assertEqual(report["rows"], 5)
        self.assertEqual(report["exact_fact_matches"], 3)
        self.assertEqual(report["disagreement_rows"], [3, 4])
        self.assertEqual(report["incomplete_rows"], [5])
        d1_result = report["per_category"]["D1"]
        self.assertEqual(d1_result["complete_pairs"], 4)
        self.assertEqual(d1_result["decision_agreement_rate"], 0.5)
        self.assertEqual(d1_result["cohen_kappa"], 0.0)
        self.assertEqual(d1_result["matrix"], {
            "pass_pass": 1, "pass_review": 1,
            "review_pass": 1, "review_review": 1,
        })
        self.assertFalse(report["reviewer_independence_verified"])
        self.assertFalse(report["label_validity_verified"])
        self.assertNotIn("source_role", json.dumps(report))

    def test_single_class_agreement_has_undefined_kappa(self):
        card = {"language": "uk", "support": {
            "applicable_fields": ["next_step"], "passed_fields": ["next_step"]}}
        stream = [record("first", card), record("second", card)]
        report = compare_evidence_batches(stream, list(reversed(stream)))
        self.assertEqual(report["language"], "uk")
        self.assertTrue(report["opaque_item_ids_matched"])
        self.assertEqual(report["per_category"]["S1"]["decision_agreement_rate"], 1.0)
        self.assertIsNone(report["per_category"]["S1"]["cohen_kappa"])

    def test_opaque_ids_prevent_false_disagreement_after_reordering(self):
        left = [record("first", d1(False)), record("second", d1(True))]
        right = [record("second", d1(True)), record("first", d1(False))]
        report = compare_evidence_batches(left, right)
        self.assertEqual(report["exact_fact_matches"], 2)
        self.assertEqual(report["per_category"]["D1"]["decision_agreements"], 2)
        self.assertEqual(report["disagreement_rows"], [])

    def test_rejects_misaligned_categories_and_languages(self):
        support = {"language": "ro", "support": {
            "applicable_fields": ["next_step"], "passed_fields": []}}
        with self.assertRaisesRegex(ValueError, "different categories"):
            compare_evidence_batches([record("one", d1(True))], [record("one", support)])
        with self.assertRaisesRegex(ValueError, "different languages"):
            compare_evidence_batches([record("one", d1(True))],
                                     [record("one", d1(True, "uk"))])
        with self.assertRaisesRegex(ValueError, "equally sized"):
            compare_evidence_batches([record("one", d1(True))], [])
        with self.assertRaisesRegex(ValueError, "same distinct opaque item IDs"):
            compare_evidence_batches([record("one", d1(True))],
                                     [record("two", d1(True))])
        with self.assertRaisesRegex(ValueError, "same distinct opaque item IDs"):
            repeated = [record("same", d1(True)), record("same", d1(False))]
            compare_evidence_batches(repeated, repeated)

    def test_cli_validates_whole_stream_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left.jsonl", Path(directory) / "right.jsonl"
            cards = [record("one", d1(False)), record("two", d1(True))]
            left.write_text("".join(json.dumps(card) + "\n" for card in cards), encoding="utf-8")
            right.write_text("".join(json.dumps(card) + "\n" for card in cards), encoding="utf-8")
            command = [sys.executable, "-m", "roguard.agreement_batch", str(left), str(right)]
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)["exact_fact_matches"], 2)
            right.write_text(right.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            invalid = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertEqual(invalid.stdout, "")


if __name__ == "__main__":
    unittest.main()
