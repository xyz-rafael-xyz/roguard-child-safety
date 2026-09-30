"""Content-free evidence agreement must abstain on conflicting declarations."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from roguard import compare_evidence

ROOT = Path(__file__).resolve().parents[1]


class AgreementTests(unittest.TestCase):
    def test_disagreement_withholds_joint_report_without_returning_values(self):
        for language in ("ro", "uk"):
            with self.subTest(language=language):
                left = json.loads((ROOT / "examples" / f"contrast_disclosure_before_{language}.json").read_text())
                right = json.loads((ROOT / "examples" / f"contrast_disclosure_after_{language}.json").read_text())
                result = compare_evidence(left, right)
                self.assertEqual(result["status"], "needs_adjudication")
                self.assertEqual(result["disagreement_paths"], ["/disclosure/safety_or_support_anchor"])
                self.assertIsNone(result["agreed_report"])
                self.assertFalse(result["reviewer_independence_verified"])
                self.assertTrue(result["review_suggested"])
                rendered = json.dumps(result, ensure_ascii=False)
                self.assertNotIn("explicit_support_request\": true", rendered)

    def test_matching_complete_and_incomplete_declarations(self):
        complete = json.loads((ROOT / "examples/contrast_disclosure_after_ro.json").read_text())
        agreed = compare_evidence(complete, complete)
        self.assertEqual(agreed["status"], "concordant")
        self.assertEqual(agreed["agreed_report"]["findings"][0]["status"], "review")
        self.assertTrue(agreed["review_suggested"])
        pending = json.loads((ROOT / "examples/reviewer_evidence_ro.json").read_text())
        unresolved = compare_evidence(pending, pending)
        self.assertEqual(unresolved["status"], "incomplete")
        self.assertTrue(unresolved["declarations_match"])
        self.assertTrue(unresolved["review_suggested"])

    def test_support_sets_ignore_order_but_require_same_members(self):
        left = {"language": "uk", "support": {
            "applicable_fields": ["next_step", "acknowledgement"],
            "passed_fields": ["acknowledgement", "next_step"]}}
        right = {"language": "uk", "support": {
            "applicable_fields": ["acknowledgement", "next_step"],
            "passed_fields": ["next_step", "acknowledgement"]}}
        self.assertEqual(compare_evidence(left, right)["status"], "concordant")
        right["support"]["passed_fields"] = ["next_step"]
        result = compare_evidence(left, right)
        self.assertEqual(result["disagreement_paths"], ["/support/passed_fields"])
        self.assertIsNone(result["agreed_report"])

    def test_rejects_unrelated_cards_and_cross_language_comparison(self):
        left = json.loads((ROOT / "examples/reviewer_evidence_ro.json").read_text())
        right = json.loads((ROOT / "examples/reviewer_evidence_uk.json").read_text())
        with self.assertRaisesRegex(ValueError, "same language"):
            compare_evidence(left, right)
        left["gate"] = {"review_owner": "person"}
        with self.assertRaisesRegex(ValueError, "D1/S1"):
            compare_evidence(left, left)

    def test_cli_reports_only_paths_for_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.json"
            right = Path(directory) / "right.json"
            left.write_bytes((ROOT / "examples/contrast_disclosure_before_ro.json").read_bytes())
            right.write_bytes((ROOT / "examples/contrast_disclosure_after_ro.json").read_bytes())
            command = [sys.executable, "-m", "roguard.agreement", str(left), str(right)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["status"], "needs_adjudication")


if __name__ == "__main__":
    unittest.main()
