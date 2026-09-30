import unittest

from roguard import (BoundaryCard, DisclosureEvidenceCard, GateCard,
                     PermissionCard, PermissionEvent, ReadabilityCard, RoutingCard,
                     SupportContractCard, check_boundary, check_disclosure_evidence,
                     check_gate, check_permission, check_readability_contract,
                     check_routing, check_support_contract)


class PolicyTests(unittest.TestCase):
    def test_declared_disclosure_facts_require_source_anchor_and_pattern(self):
        card = DisclosureEvidenceCard("minor", True, True, False)
        self.assertEqual(check_disclosure_evidence(card).reason_codes, ("DECLARED_INDIRECT_SUPPORT",))
        self.assertEqual(check_disclosure_evidence(DisclosureEvidenceCard("adult", True, True, False)).reason_codes, ())
        self.assertEqual(check_disclosure_evidence(DisclosureEvidenceCard("minor", False, True, False)).reason_codes, ())
        self.assertEqual(check_disclosure_evidence(DisclosureEvidenceCard("minor", True, None, False)).reason_codes,
                         ("MISSING_SIGNAL_EVIDENCE",))
        with self.assertRaisesRegex(ValueError, "booleans or null"):
            check_disclosure_evidence(DisclosureEvidenceCard("minor", "yes", True, False))

    def test_support_contract_attributes_each_failed_declared_field(self):
        card = SupportContractCard(frozenset({"next_step", "no_secret_promise", "human_support"}),
                                   frozenset({"human_support"}))
        self.assertEqual(check_support_contract(card).reason_codes,
                         ("SUPPORT_NEXT_STEP_FAILED", "SUPPORT_NO_SECRET_PROMISE_FAILED"))
        self.assertEqual(check_support_contract(SupportContractCard(frozenset(), frozenset())).reason_codes,
                         ("MISSING_SUPPORT_CONTRACT",))
        with self.assertRaisesRegex(ValueError, "known and applicable"):
            check_support_contract(SupportContractCard(frozenset({"next_step"}), frozenset({"unknown"})))
        with self.assertRaisesRegex(ValueError, "frozensets"):
            check_support_contract(SupportContractCard({"next_step"}, frozenset()))

    def test_word_cap_contract_does_not_invent_an_age_threshold(self):
        self.assertEqual(check_readability_contract(ReadabilityCard(9, 76, 75)).reason_codes,
                         ("WORD_CAP_EXCEEDED",))
        self.assertFalse(check_readability_contract(ReadabilityCard(9, 75, 75)).review_suggested)
        self.assertEqual(check_readability_contract(ReadabilityCard(9, 30, None)).reason_codes,
                         ("MISSING_CONTRACT",))
        with self.assertRaisesRegex(ValueError, "negative"):
            check_readability_contract(ReadabilityCard(9, -1, 75))

    def test_guardian_has_no_implicit_access(self):
        card = RoutingCard("minor", "item-x", "purpose-a", "guardian", frozenset({"minor", "guardian"}),
                           {("minor", "item-x", "purpose-a"): frozenset({"minor"})})
        self.assertEqual(check_routing(card).reason_codes, ("PRINCIPAL_SCOPE_CONFLICT",))
        other_person = RoutingCard("minor", "item-x", "purpose-a", "guardian",
                                   frozenset({"minor", "guardian"}),
                                   {("another-principal", "item-x", "purpose-a"): frozenset({"guardian"})})
        self.assertEqual(check_routing(other_person).reason_codes, ("MISSING_POLICY",))
        self.assertEqual(check_routing(RoutingCard(None, None, None, "guardian", None, None)).reason_codes,
                         ("MISSING_POLICY", "MISSING_PURPOSE", "MISSING_ROLE"))

    def test_direct_api_rejects_string_membership_false_pass(self):
        # Before validation, "guard" in "guardian" passed both membership checks.
        malformed = RoutingCard("minor", "item", "purpose", "guard", "guardian",
                                {("minor", "item", "purpose"): "guardian"})
        with self.assertRaisesRegex(ValueError, "frozensets"):
            check_routing(malformed)
        malformed_scope = RoutingCard("minor", "item", "purpose", "guard",
                                      frozenset({"guard"}),
                                      {("minor", "item", "purpose"): "guardian"})
        with self.assertRaisesRegex(ValueError, "frozensets"):
            check_routing(malformed_scope)
        valid = RoutingCard("minor", "item", "purpose", "guard", frozenset({"guard"}),
                            {("minor", "item", "purpose"): frozenset({"guardian"})})
        self.assertEqual(check_routing(valid).reason_codes, ("PRINCIPAL_SCOPE_CONFLICT",))

    def test_direct_api_rejects_malformed_facts_before_incomplete_or_pass(self):
        with self.assertRaises(ValueError):
            check_disclosure_evidence(DisclosureEvidenceCard([], None, None, None))
        with self.assertRaises(ValueError):
            check_support_contract(SupportContractCard(frozenset(), "next_step"))
        with self.assertRaises(ValueError):
            check_readability_contract(ReadabilityCard(True, False, 1))
        with self.assertRaises(ValueError):
            check_permission(PermissionCard("minor", "item", "purpose", "recipient", 2,
                                            0, (PermissionEvent(1, "minor", "item", "purpose",
                                                                "recipient", "grant"),), False))
        with self.assertRaises(ValueError):
            check_permission(PermissionCard(None, None, None, None, None, False,
                                            (PermissionEvent(1, "minor", "item", "purpose",
                                                             "recipient", "invalid"),)))
        with self.assertRaises(ValueError):
            check_boundary(BoundaryCard(None, None, None, None, 0, False))
        with self.assertRaises(ValueError):
            check_gate(GateCard("reviewer", True, False, "", False))

    def test_whitespace_only_authority_is_missing(self):
        self.assertEqual(check_gate(GateCard("   ", True, False, True, True)).reason_codes,
                         ("MISSING_REVIEW_OWNER",))
        route = RoutingCard("   ", "item", "purpose", "minor", frozenset({"minor"}),
                            {("minor", "item", "purpose"): frozenset({"minor"})})
        self.assertEqual(check_routing(route).reason_codes, ("MISSING_ROLE",))
        permission = PermissionCard("  ", "item", "purpose", "minor", 2, True,
                                    (PermissionEvent(1, "minor", "item", "purpose", "minor", "grant"),))
        self.assertEqual(check_permission(permission).reason_codes, ("MISSING_STATE",))
        with self.assertRaises(ValueError):
            check_routing(RoutingCard("minor", "item", "purpose", " ", frozenset({" "}), {}))

    def test_revocation_and_new_grant_are_distinct(self):
        events = (
            PermissionEvent(1, "minor", "item-x", "purpose-a", "recipient-y", "grant"),
            PermissionEvent(2, "minor", "item-x", "purpose-a", "recipient-y", "revoke"),
        )
        card = PermissionCard("minor", "item-x", "purpose-a", "recipient-y", 3, True, events)
        self.assertEqual(check_permission(card).reason_codes, ("UNAUTHORIZED_REUSE",))
        renewed = events + (PermissionEvent(4, "minor", "item-x", "purpose-a", "recipient-y", "grant"),)
        card = PermissionCard("minor", "item-x", "purpose-a", "recipient-y", 5, False, renewed)
        self.assertEqual(check_permission(card).reason_codes, ("OVER_WITHHOLD",))
        optional = PermissionCard("minor", "item-x", "purpose-a", "recipient-y", 5, False, renewed,
                                  use_required=False)
        self.assertEqual(check_permission(optional).reason_codes, ())

    def test_boundary_recurrence_and_utility_can_both_fail(self):
        card = BoundaryCard(frozenset({"needed"}), frozenset({"protected"}),
                            frozenset({"needed"}), frozenset({"protected"}), True, True)
        self.assertEqual(check_boundary(card).reason_codes,
                         ("BOUNDARY_RECURRENCE", "OVER_WITHHOLD"))
        new_field = BoundaryCard(frozenset({"needed"}), frozenset({"protected"}),
                                 frozenset(), frozenset({"unlisted"}), True, True)
        self.assertEqual(check_boundary(new_field).reason_codes, ("UNAUTHORIZED_REUSE",))
        both = BoundaryCard(frozenset({"needed"}), frozenset({"protected"}),
                            frozenset(), frozenset({"protected", "unlisted"}), True, True)
        self.assertEqual(check_boundary(both).reason_codes,
                         ("BOUNDARY_RECURRENCE", "UNAUTHORIZED_REUSE"))

    def test_gate_flags_fail_open_and_automatic_branch(self):
        card = GateCard("reviewer", False, True, True, False)
        self.assertEqual(check_gate(card).reason_codes, ("FAIL_OPEN", "AUTO_EXTERNAL_ACTION"))
        safe = GateCard("reviewer", False, False, False, True)
        self.assertFalse(check_gate(safe).review_suggested)
        self.assertFalse(check_gate(GateCard("reviewer", True, False, False, True)).review_suggested)
        future_fail_open = GateCard("reviewer", True, True, False, True)
        self.assertEqual(check_gate(future_fail_open).reason_codes, ("FAIL_OPEN",))


if __name__ == "__main__":
    unittest.main()
