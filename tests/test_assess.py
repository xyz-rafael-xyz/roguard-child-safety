import unittest

from roguard import (GateCard, ReadabilityCard, RoutingCard, ScreenResult,
                     assemble_report, assess_contracts)


class ContractAssessmentTests(unittest.TestCase):
    def test_independent_findings_preserve_both_failures(self):
        routing = RoutingCard(
            "minor", "item", "A", "guardian", frozenset({"minor", "guardian"}),
            {("minor", "item", "A"): frozenset({"minor"})},
        )
        result = assess_contracts(
            readability=ReadabilityCard(9, 76, 75), routing=routing,
            gate=GateCard("reviewer", True, False, False, True),
        )
        self.assertEqual([(item.category, item.status) for item in result.findings],
                         [("A1", "review"), ("R1", "review"), ("G1", "pass")])
        self.assertTrue(result.review_suggested)
        self.assertFalse(result.incomplete)

    def test_missing_policy_stays_incomplete(self):
        result = assess_contracts(routing=RoutingCard(None, None, None, "guardian", None, None))
        self.assertEqual(result.findings[0].status, "incomplete")
        self.assertTrue(result.incomplete)
        self.assertTrue(result.review_suggested)

    def test_explicit_bypass_takes_priority_over_missing_owner(self):
        result = assess_contracts(gate=GateCard(None, True, False, True, False))
        self.assertEqual(result.findings[0].status, "review")
        self.assertEqual(result.findings[0].reason_codes,
                         ("MISSING_REVIEW_OWNER", "AUTO_EXTERNAL_ACTION"))

    def test_unparsed_output_is_visible_as_blocked_state(self):
        result = assess_contracts(gate=GateCard("reviewer", False, False, False, True))
        self.assertEqual(result.findings[0].status, "incomplete")
        self.assertEqual(result.findings[0].reason_codes, ("OUTPUT_BLOCKED",))
        self.assertTrue(result.review_suggested)

    def test_empty_assessment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            assess_contracts()

    def test_model_cannot_replace_a_routing_policy(self):
        signal = ScreenResult("ro", "routing_card", {"R1": 1.0}, ("R1",))
        report = assemble_report(model_signal=signal)
        self.assertEqual(report.unverified_policy_labels, ("R1",))
        self.assertEqual(report.advisory_labels, ())
        self.assertTrue(report.review_suggested)
        contracts = assess_contracts(routing=RoutingCard(None, None, None, "guardian", None, None))
        report = assemble_report(contracts=contracts, model_signal=signal)
        self.assertTrue(report.review_suggested)
        self.assertTrue(report.contracts.incomplete)

    def test_complete_declared_route_takes_precedence_over_model_policy_label(self):
        signal = ScreenResult("ro", "routing_card", {"R1": 1.0}, ("R1",))
        route = RoutingCard("minor", "item", "A", "minor", frozenset({"minor", "guardian"}),
                            {("minor", "item", "A"): frozenset({"minor"})})
        report = assemble_report(contracts=assess_contracts(routing=route), model_signal=signal)
        self.assertEqual(report.contracts.findings[0].status, "pass")
        self.assertEqual(report.unverified_policy_labels, ("R1",))
        self.assertEqual(report.policy_disagreements, ("R1",))
        self.assertEqual(report.policy_labels_without_contract, ())
        self.assertFalse(report.review_suggested)

    def test_linguistic_signal_is_advisory(self):
        signal = ScreenResult("ro", "message", {"D1": 1.0}, ("D1",))
        report = assemble_report(model_signal=signal)
        self.assertEqual(report.advisory_labels, ("D1",))
        self.assertTrue(report.review_suggested)

    def test_invalid_model_signal_cannot_cross_source_kinds(self):
        signal = ScreenResult("ro", "routing_card", {"D1": 1.0}, ("D1",))
        with self.assertRaisesRegex(ValueError, "invalid language, category"):
            assemble_report(model_signal=signal)

    def test_inactive_language_signal_is_rejected(self):
        signal = ScreenResult("uk", "message", {"D1": 1.0}, ("D1",))
        with self.assertRaisesRegex(ValueError, "invalid language, category"):
            assemble_report(model_signal=signal)


if __name__ == "__main__":
    unittest.main()
