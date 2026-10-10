"""Decision stability must be exact for finite unknowns and honest otherwise."""

import json
import unittest
from pathlib import Path

from jsonschema import validate

from roguard.v1_envelope import envelope_v1_contract


class V1EnvelopeTests(unittest.TestCase):
    def test_report_matches_public_schema(self):
        schema = json.loads((Path(__file__).resolve().parents[1] / "schema" /
                             "v1-envelope-report.schema.json").read_text(encoding="utf-8"))
        for language in ("ro", "uk"):
            report = envelope_v1_contract({"language": language, "support": {
                "applicable_fields": ["next_step"], "passed_fields": None}})
            validate(report, schema)

    def test_d1_direct_signal_is_stable_despite_retraction_and_missing_fields(self):
        for language in ("ro", "uk"):
            report = envelope_v1_contract({"language": language, "disclosure": {
                "source_role": "minor", "safety_or_support_anchor": None,
                "indirect_or_repeated_support_pattern": None,
                "explicit_support_request": None,
                "direct_safety_statement": True, "retracted_or_denied": None}})
            row = report["categories"][0]
            self.assertEqual(row["completion_count"], 16)
            self.assertEqual(row["decision_stability"], "stable")
            self.assertEqual(row["possible_decision_codes"], ["REVIEW_SUPPORT_SIGNAL"])
            self.assertTrue(report["caller_facts_unverified"])
            self.assertFalse(report["external_action_taken"])

    def test_d1_unknown_role_exposes_both_decisions(self):
        report = envelope_v1_contract({"language": "ro", "disclosure": {
            "source_role": None, "safety_or_support_anchor": False,
            "indirect_or_repeated_support_pattern": False,
            "explicit_support_request": True, "direct_safety_statement": False}})
        row = report["categories"][0]
        self.assertEqual(row["completion_count"], 6)
        self.assertEqual(row["decision_stability"], "sensitive")
        self.assertEqual(row["possible_decision_codes"],
                         ["NO_D1_SIGNAL", "REVIEW_SUPPORT_SIGNAL"])

    def test_r1_concern_uncertainty_and_unbounded_policy_are_distinct(self):
        routing = {"principal": "minor", "item": "private-marker", "purpose": "A",
                   "proposed_recipient": "custom-adult", "recipient_roles": ["custom-adult"],
                   "allowed_by_scope": [{"principal": "minor", "item": "private-marker",
                                         "purpose": "A", "recipients": ["custom-adult"]}],
                   "recipient_may_be_source_of_concern": "unknown"}
        report = envelope_v1_contract({"language": "uk", "routing": routing})
        row = report["categories"][0]
        self.assertEqual(row["possible_decision_codes"], ["NO_ROUTE_SIGNAL", "REVIEW_ROUTE"])
        self.assertEqual(row["completion_count"], 2)
        self.assertNotIn("private-marker", json.dumps(report))
        self.assertNotIn("custom-adult", json.dumps(report))
        routing["allowed_by_scope"] = None
        row = envelope_v1_contract({"language": "uk", "routing": routing})["categories"][0]
        self.assertEqual(row["decision_stability"], "unbounded")
        self.assertIsNone(row["completion_count"])
        self.assertEqual(row["possible_decision_codes"], [])

    def test_s1_unknown_applicability_enumerates_valid_completions(self):
        report = envelope_v1_contract({"language": "ro", "support": {
            "applicable_fields": None, "passed_fields": ["next_step"]}})
        row = report["categories"][0]
        self.assertEqual(row["completion_count"], 3 ** 9)
        self.assertEqual(row["possible_decision_codes"],
                         ["NO_RESPONSE_SIGNAL", "REVIEW_RESPONSE"])
        report = envelope_v1_contract({"language": "uk", "support": {
            "applicable_fields": [], "passed_fields": None}})
        row = report["categories"][0]
        self.assertEqual(row["completion_count"], 1)
        self.assertEqual(row["possible_decision_codes"], ["INSUFFICIENT_RESPONSE_CONTRACT"])

    def test_invalid_card_is_rejected(self):
        with self.assertRaises(ValueError):
            envelope_v1_contract({"language": "ro", "support": {
                "applicable_fields": ["unknown"], "passed_fields": []}})


if __name__ == "__main__":
    unittest.main()
