"""Local enumerated D1 review keeps unknown facts and avoids partial output."""

import json
import subprocess
import sys
import unittest

from roguard import collect_disclosure_card, collect_support_card, compare_evidence
from roguard.cli import assess_json


class ReviewCardTests(unittest.TestCase):
    def test_s1_numeric_form_records_field_specific_failure_in_both_languages(self):
        for language in ("ro", "uk"):
            with self.subTest(language=language):
                answers = iter(("1", "1", "1", "2", "2", "2", "2", "2", "2", "2"))
                prompts = []
                card = collect_support_card(language, answers.__next__, prompts.append)
                self.assertEqual(card["support"], {
                    "applicable_fields": ["acknowledgement", "next_step"],
                    "passed_fields": ["acknowledgement"],
                })
                finding = assess_json(card)["findings"][0]
                self.assertEqual(finding["status"], "review")
                self.assertIn("SUPPORT_NEXT_STEP_FAILED", finding["reason_codes"])
                self.assertEqual(compare_evidence(card, card)["status"], "concordant")
                self.assertEqual(len(prompts), 11)

    def test_s1_unknown_applicability_stays_incomplete(self):
        card = collect_support_card("ro", iter(("3", *(["2"] * 7))).__next__, lambda _: None)
        self.assertEqual(card["support"], {"applicable_fields": None, "passed_fields": None})
        self.assertEqual(assess_json(card)["findings"][0]["status"], "incomplete")

    def test_s1_invalid_input_produces_no_partial_stdout(self):
        run = subprocess.run(
            [sys.executable, "-m", "roguard.review_card", "--language", "uk",
             "--category", "S1", "--compact", "--item-id", "abstract_s1"],
            input="source text\ninvalid\n4\n", text=True, capture_output=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(run.stdout, "")
        self.assertNotIn("source text", run.stderr)

    def test_bilingual_cards_preserve_unknown_and_support_agreement(self):
        for language in ("ro", "uk"):
            with self.subTest(language=language):
                prompts = []
                answers = iter(("1\n", "1\n", "2\n", "3\n"))
                card = collect_disclosure_card(language, lambda: next(answers), prompts.append)
                self.assertEqual(card["disclosure"], {
                    "source_role": "minor", "safety_or_support_anchor": True,
                    "indirect_or_repeated_support_pattern": False,
                    "explicit_support_request": None,
                })
                self.assertEqual(assess_json(card)["findings"][0]["status"], "incomplete")
                self.assertEqual(compare_evidence(card, card)["status"], "incomplete")
                self.assertEqual(len(prompts), 5)

    def test_one_fact_disagreement_withholds_joint_finding(self):
        before = collect_disclosure_card("ro", iter(("1", "2", "1", "2")).__next__, lambda _: None)
        after = collect_disclosure_card("ro", iter(("1", "1", "1", "2")).__next__, lambda _: None)
        report = compare_evidence(before, after)
        self.assertEqual(report["status"], "needs_adjudication")
        self.assertEqual(report["disagreement_paths"], ["/disclosure/safety_or_support_anchor"])
        self.assertIsNone(report["agreed_report"])

    def test_invalid_choice_is_not_echoed_and_cannot_emit_partial_card(self):
        seen = []
        answers = iter(("unexpected free text\n", "5\n", "not-an-option\n"))
        with self.assertRaises(ValueError):
            collect_disclosure_card("ro", answers.__next__, seen.append)
        self.assertNotIn("unexpected free text", " ".join(seen))
        self.assertNotIn("not-an-option", " ".join(seen))
        self.assertEqual(sum("Alegeți" in item for item in seen), 3)

    def test_cli_prints_only_complete_card_to_stdout(self):
        run = subprocess.run(
            [sys.executable, "-m", "roguard.review_card", "--language", "uk"],
            input="1\n1\n2\n1\n", text=True, capture_output=True, check=True)
        card = json.loads(run.stdout)
        self.assertEqual(card["language"], "uk")
        self.assertEqual(assess_json(card)["findings"][0]["status"], "review")
        self.assertIn("Картка D1", run.stderr)
        incomplete = subprocess.run(
            [sys.executable, "-m", "roguard.review_card", "--language", "uk"],
            input="1\n", text=True, capture_output=True)
        self.assertNotEqual(incomplete.returncode, 0)
        self.assertEqual(incomplete.stdout, "")

    def test_compact_cli_emits_one_jsonl_row(self):
        run = subprocess.run(
            [sys.executable, "-m", "roguard.review_card", "--language", "ro",
             "--compact", "--item-id", "abstract_01"],
            input="1\n2\n1\n2\n", text=True, capture_output=True, check=True)
        self.assertEqual(len(run.stdout.splitlines()), 1)
        row = json.loads(run.stdout)
        self.assertEqual(row["item_id"], "abstract_01")
        self.assertEqual(row["card"]["disclosure"]["safety_or_support_anchor"], False)


if __name__ == "__main__":
    unittest.main()
