"""Finite symbolic truth tables independent of model and generator surfaces."""

import itertools
import unittest

from roguard.assess import assess_contracts
from roguard.policy import (BoundaryCard, DisclosureEvidenceCard, GateCard,
                            ReadabilityCard, RoutingCard, SUPPORT_FIELDS,
                            SupportContractCard, check_boundary,
                            check_disclosure_evidence, check_readability_contract,
                            check_routing, check_support_contract)


class ContractMatrixTests(unittest.TestCase):
    def test_disclosure_pattern_truth_table(self):
        checked = 0
        for role, anchor, indirect, explicit in itertools.product(
            (None, "minor", "adult", "other"), (None, False, True),
            (None, False, True), (None, False, True)
        ):
            if None in (role, anchor, indirect, explicit):
                expected = ("MISSING_SIGNAL_EVIDENCE",)
            elif role == "minor" and anchor and explicit:
                expected = ("DECLARED_EXPLICIT_SUPPORT",)
            elif role == "minor" and anchor and indirect:
                expected = ("DECLARED_INDIRECT_SUPPORT",)
            else:
                expected = ()
            got = check_disclosure_evidence(DisclosureEvidenceCard(role, anchor, indirect, explicit))
            self.assertEqual(got.reason_codes, expected)
            checked += 1
        self.assertEqual(checked, 108)

    def test_word_cap_numerical_matrix(self):
        checked = 0
        for age, cap, count in itertools.product((0, 7, 12), range(13), range(15)):
            expected = ("WORD_CAP_EXCEEDED",) if count > cap else ()
            got = check_readability_contract(ReadabilityCard(age, count, cap))
            self.assertEqual(got.reason_codes, expected)
            checked += 1
        self.assertEqual(checked, 585)

    def test_routing_scope_matrix(self):
        roles = frozenset({"minor", "guardian"})
        checked = 0
        for principal, item, purpose, recipient, allowed in itertools.product(
            ("minor", "other"), ("item-a", "item-b"), ("p", "q"),
            ("minor", "guardian", "stranger"),
            (frozenset(), frozenset({"minor"}), frozenset({"guardian"}), roles),
        ):
            scope = {("minor", "item-a", "p"): allowed}
            card = RoutingCard(principal, item, purpose, recipient, roles, scope)
            if recipient not in roles:
                expected = ("PRINCIPAL_SCOPE_CONFLICT",)
            elif (principal, item, purpose) not in scope:
                expected = ("MISSING_POLICY",)
            elif recipient not in allowed:
                expected = ("PRINCIPAL_SCOPE_CONFLICT",)
            else:
                expected = ()
            self.assertEqual(check_routing(card).reason_codes, expected)
            checked += 1
        self.assertEqual(checked, 96)

    def test_boundary_preservation_matrix(self):
        available = ("required", "optional", "protected", "unlisted")
        checked = 0
        for bits, corrected, renewed in itertools.product(
            itertools.product((False, True), repeat=4), (False, True), (False, True)
        ):
            proposed = frozenset(field for field, included in zip(available, bits) if included)
            expected = []
            if "protected" in proposed:
                expected.append("BOUNDARY_RECURRENCE" if corrected and renewed else "UNAUTHORIZED_REUSE")
            if "unlisted" in proposed and "UNAUTHORIZED_REUSE" not in expected:
                expected.append("UNAUTHORIZED_REUSE")
            if "required" not in proposed:
                expected.append("OVER_WITHHOLD")
            card = BoundaryCard(frozenset({"required", "optional"}),
                                frozenset({"protected"}), frozenset({"required"}),
                                proposed, corrected, renewed)
            self.assertEqual(check_boundary(card).reason_codes, tuple(expected))
            checked += 1
        self.assertEqual(checked, 64)

    def test_gate_branches_and_incomplete_status(self):
        checked = 0
        for owner, parsed, accept_invalid, external, approval in itertools.product(
            (None, "reviewer"), (False, True), (False, True), (False, True), (False, True)
        ):
            expected = []
            if owner is None:
                expected.append("MISSING_REVIEW_OWNER")
            if accept_invalid:
                expected.append("FAIL_OPEN")
            if external and not approval:
                expected.append("AUTO_EXTERNAL_ACTION")
            if not parsed and not accept_invalid:
                expected.append("OUTPUT_BLOCKED")
            card = GateCard(owner, parsed, accept_invalid, external, approval)
            finding = assess_contracts(gate=card).findings[0]
            self.assertEqual(finding.reason_codes, tuple(expected))
            status = ("pass" if not expected else "incomplete" if
                      set(expected) <= {"MISSING_REVIEW_OWNER", "OUTPUT_BLOCKED"} else "review")
            self.assertEqual(finding.status, status)
            checked += 1
        self.assertEqual(checked, 32)

    def test_support_field_matrix(self):
        fields = sorted(SUPPORT_FIELDS)
        checked = 0
        for states in itertools.product((0, 1, 2), repeat=len(fields)):
            applicable = frozenset(field for field, state in zip(fields, states) if state)
            passed = frozenset(field for field, state in zip(fields, states) if state == 1)
            expected = (("MISSING_SUPPORT_CONTRACT",) if not applicable else
                        tuple(f"SUPPORT_{field.upper()}_FAILED" for field, state in
                              zip(fields, states) if state == 2))
            got = check_support_contract(SupportContractCard(applicable, passed))
            self.assertEqual(got.reason_codes, expected)
            checked += 1
        self.assertEqual(checked, 6561)


if __name__ == "__main__":
    unittest.main()
