"""V2 audit decisions must match the reviewed symbolic rule engine."""

import json
import random
import unittest
from itertools import product
from pathlib import Path

from jsonschema import validate

from roguard.v1_contract import assess_v1_json
from roguard.v2_audit import audit_v2_contract


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("acknowledgement", "next_step", "human_support", "no_blame",
          "no_secret_promise", "no_leading_question", "no_unsupported_guarantee",
          "no_dependency_pressure", "immediate_danger_route", "no_pressing_for_details")
STATES = ("not_applicable", "failed", "passed")


def _v1_support(states):
    return {"applicable_fields": [field for field, state in states.items()
                                  if state != "not_applicable"],
            "passed_fields": [field for field, state in states.items()
                              if state == "passed"]}


class V2AuditTests(unittest.TestCase):
    def test_schema_and_exact_full_support_space(self):
        payload = {"language": "ro", "support": {"field_states": {}}}
        report = audit_v2_contract(payload)
        validate(payload, json.loads((ROOT / "schema/v2-audit-input.schema.json").read_text()))
        validate(report, json.loads((ROOT / "schema/v2-audit-report.schema.json").read_text()))
        row = report["categories"][0]
        self.assertEqual(row["completion_count"], 3 ** len(FIELDS))
        self.assertEqual(set(row["possible_decision_codes"]),
                         {"INSUFFICIENT_RESPONSE_CONTRACT", "NO_RESPONSE_SIGNAL", "REVIEW_RESPONSE"})
        self.assertEqual(len(row["witnesses"]), 3)
        self.assertEqual(len(row["unknown_paths"]), len(FIELDS))
        self.assertFalse(report["external_action_taken"])

    def test_partial_states_match_v1_oracle_and_narrow_when_resolved(self):
        rng = random.Random(20261011)
        for language in ("ro", "uk"):
            for _ in range(30):
                given = {field: rng.choice(STATES) for field in FIELDS}
                unknown = rng.sample(FIELDS, rng.randrange(1, 4))
                for field in unknown:
                    given[field] = "unknown"
                payload = {"language": language, "support": {"field_states": given}}
                row = audit_v2_contract(payload)["categories"][0]
                expected = set()
                for answers in product(STATES, repeat=len(unknown)):
                    filled = dict(given)
                    filled.update(zip(unknown, answers))
                    expected.add(assess_v1_json({"language": language,
                                                "support": _v1_support(filled)})["checks"][0]["decision_code"])
                self.assertEqual(set(row["possible_decision_codes"]), expected)
                self.assertEqual(row["completion_count"], 3 ** len(unknown))
                self.assertEqual(len(row["witnesses"]), len(expected))
                for witness in row["witnesses"]:
                    filled = dict(given)
                    for fact in witness["unknown_facts"]:
                        filled[fact["path"].rsplit("/", 1)[-1]] = fact["state_code"]
                    decision = assess_v1_json({"language": language,
                                               "support": _v1_support(filled)})["checks"][0]["decision_code"]
                    self.assertEqual(decision, witness["decision_code"])
                resolved = dict(given)
                resolved[unknown[0]] = STATES[0]
                narrower = audit_v2_contract({"language": language,
                                              "support": {"field_states": resolved}})["categories"][0]
                self.assertLessEqual(set(narrower["possible_decision_codes"]), expected)
                self.assertEqual(narrower["completion_count"], row["completion_count"] // 3)

    def test_question_ranking_identifies_decisive_support_fact(self):
        states = {field: "not_applicable" for field in FIELDS}
        states["next_step"] = "passed"
        states["no_pressing_for_details"] = "unknown"
        row = audit_v2_contract({"language": "uk", "support": {
            "field_states": states}})["categories"][0]
        self.assertEqual(row["next_questions"][0]["path"],
                         "/support/field_states/no_pressing_for_details")
        self.assertEqual(row["next_questions"][0]["worst_case_decision_count"], 1)
        self.assertEqual({branch["answer_code"]: branch["possible_decision_codes"]
                          for branch in row["next_questions"][0]["branches"]},
                         {"failed": ["REVIEW_RESPONSE"],
                          "not_applicable": ["NO_RESPONSE_SIGNAL"],
                          "passed": ["NO_RESPONSE_SIGNAL"]})
        self.assertEqual(row["completion_count"], 3)

    def test_disclosure_and_routing_witnesses_have_no_custom_values(self):
        payload = {"language": "ro", "disclosure": {
            "source_role": None, "safety_or_support_anchor": False,
            "indirect_or_repeated_support_pattern": False,
            "explicit_support_request": True, "direct_safety_statement": False,
            "retracted_or_denied": True},
            "routing": {"principal": "minor", "item": "secret-marker", "purpose": "P",
                        "proposed_recipient": "custom-adult",
                        "recipient_roles": ["custom-adult"],
                        "allowed_by_scope": [{"principal": "minor", "item": "secret-marker",
                                              "purpose": "P", "recipients": ["custom-adult"]}],
                        "recipient_may_be_source_of_concern": "unknown"}}
        report = audit_v2_contract(payload)
        self.assertEqual([row["decision_stability"] for row in report["categories"]],
                         ["sensitive", "sensitive"])
        self.assertEqual(report["categories"][0]["next_questions"][0]["path"],
                         "/disclosure/source_role")
        self.assertEqual(report["categories"][1]["next_questions"][0]["path"],
                         "/routing/recipient_may_be_source_of_concern")
        self.assertNotIn("secret-marker", json.dumps(report))
        self.assertNotIn("custom-adult", json.dumps(report))
        validate(report, json.loads((ROOT / "schema/v2-audit-report.schema.json").read_text()))

    def test_unbounded_route_and_invalid_states(self):
        payload = {"language": "uk", "routing": {"principal": "minor", "item": "x",
                   "purpose": "A", "proposed_recipient": "r", "recipient_roles": ["r"],
                   "allowed_by_scope": None, "recipient_may_be_source_of_concern": "unknown"}}
        row = audit_v2_contract(payload)["categories"][0]
        self.assertEqual(row["decision_stability"], "unbounded")
        self.assertIsNone(row["completion_count"])
        self.assertEqual(row["witnesses"], [])
        for states in ({"unknown_field": "passed"}, {"next_step": "mystery"},
                       {"next_step": 1}):
            with self.assertRaises(ValueError):
                audit_v2_contract({"language": "ro", "support": {"field_states": states}})

    def test_v1_rejects_impossible_known_support_state(self):
        with self.assertRaises(ValueError):
            assess_v1_json({"language": "ro", "support": {
                "applicable_fields": [], "passed_fields": ["next_step"]}})


if __name__ == "__main__":
    unittest.main()
