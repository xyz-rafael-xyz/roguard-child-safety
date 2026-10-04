import json
import itertools
import unittest
from pathlib import Path

from roguard.policy import RoutingCard, SupportContractCard
from roguard.policy_v1 import (DisclosureCardV1, RoutingCardV1, check_disclosure_v1,
                               check_routing_v1, check_support_v1)
from roguard.v1_contract import assess_v1_json

ROOT = Path(__file__).resolve().parents[1]


class V1ContractTests(unittest.TestCase):
    def test_direct_statement_survives_retraction(self):
        card = DisclosureCardV1("minor", True, False, False, True, True)
        self.assertEqual(check_disclosure_v1(card).reason_codes,
                         ("DECLARED_DIRECT_SAFETY_STATEMENT",))
        self.assertEqual(check_disclosure_v1(DisclosureCardV1("minor", False, False, False, True)).reason_codes,
                         ("DECLARED_DIRECT_SAFETY_STATEMENT",))
        self.assertEqual(check_disclosure_v1(DisclosureCardV1("minor", None, None, False, True)).reason_codes,
                         ("DECLARED_DIRECT_SAFETY_STATEMENT",))
        self.assertEqual(check_disclosure_v1(DisclosureCardV1("minor", True, False, False, False)).reason_codes, ())

    def test_retraction_never_changes_d1_route_across_declared_flags(self):
        for anchor, indirect, request, statement in itertools.product((True, False, None), repeat=4):
            without = check_disclosure_v1(DisclosureCardV1(
                "minor", anchor, indirect, request, statement, False))
            with_retraction = check_disclosure_v1(DisclosureCardV1(
                "minor", anchor, indirect, request, statement, True))
            self.assertEqual(without.reason_codes, with_retraction.reason_codes)
            if statement:
                self.assertEqual(without.reason_codes, ("DECLARED_DIRECT_SAFETY_STATEMENT",))

    def test_unknown_recipient_concern_never_aligns(self):
        old = RoutingCard("minor", "symbol", "A", "adult", frozenset({"adult"}),
                          {("minor", "symbol", "A"): frozenset({"adult"})})
        self.assertEqual(check_routing_v1(RoutingCardV1(old, "no")).reason_codes, ())
        for concern in ("yes", "unknown"):
            self.assertIn("SOURCE_OF_CONCERN_UNRESOLVED",
                          check_routing_v1(RoutingCardV1(old, concern)).reason_codes)

    def test_conditional_support_fields(self):
        self.assertEqual(check_support_v1(SupportContractCard(None, None)).reason_codes,
                         ("INSUFFICIENT_RESPONSE_CONTRACT",))
        self.assertEqual(check_support_v1(SupportContractCard(
            frozenset({"no_pressing_for_details"}), frozenset())).reason_codes,
            ("SUPPORT_NO_PRESSING_FOR_DETAILS_FAILED",))
        self.assertEqual(check_support_v1(SupportContractCard(
            frozenset({"next_step"}), frozenset({"next_step"}))).reason_codes, ())

    def test_examples_and_reject_unknown_fields(self):
        for language in ("ro", "uk"):
            payload = json.loads((ROOT / f"examples/v1_candidate_{language}.json").read_text(encoding="utf-8"))
            report = assess_v1_json(payload)
            self.assertEqual([check["category"] for check in report["checks"]], ["D1", "R1", "S1"])
            self.assertFalse(report["external_action_taken"])
        payload["disclosure"]["raw_child_message"] = "forbidden"
        with self.assertRaises(ValueError):
            assess_v1_json(payload)


if __name__ == "__main__":
    unittest.main()
