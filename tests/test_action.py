"""A proposed use needs consistent routing, permission, and workflow facts."""

import copy
import unittest

from roguard import assess_proposed_use
from roguard.cli import assess_json, assess_jsonl
from roguard.contrast import contrast_declared_contracts
from roguard.policy import GateCard, PermissionCard, PermissionEvent, RoutingCard


def card_payload() -> dict:
    return {
        "language": "ro",
        "routing": {
            "principal": "minor", "item": "symbol", "purpose": "A",
            "proposed_recipient": "reader",
            "recipient_roles": ["reader"],
            "allowed_by_scope": [{"principal": "minor", "item": "symbol",
                                  "purpose": "A", "recipients": ["reader"]}],
        },
        "permission": {
            "principal": "minor", "item": "symbol", "purpose": "A",
            "recipient": "reader", "at": 2, "proposed_use": True,
            "events": [{"sequence": 1, "principal": "minor", "item": "symbol",
                        "purpose": "A", "recipient": "reader", "action": "grant"}],
            "use_required": False,
        },
        "proposed_use": {"external_action": False},
    }


class ProposedUseTests(unittest.TestCase):
    def test_matching_cards_are_only_ready_for_human_decision(self):
        for language in ("ro", "uk"):
            payload = card_payload()
            payload["language"] = language
            result = assess_json(payload)
            self.assertEqual(result["proposed_use"], {
                "status": "ready_for_human_decision", "reason_codes": [],
                "incomplete": False,
                "action_taken": False, "caller_facts_verified": False,
            })
            self.assertTrue(result["review_suggested"])

    def test_individually_passing_cards_must_share_one_scope(self):
        payload = card_payload()
        payload["permission"]["purpose"] = "B"
        payload["permission"]["events"][0]["purpose"] = "B"
        result = assess_json(payload)
        self.assertTrue(all(item["status"] == "pass" for item in result["findings"]))
        self.assertEqual(result["proposed_use"]["status"], "review")
        self.assertEqual(result["proposed_use"]["reason_codes"], ["SCOPE_MISMATCH"])
        self.assertTrue(result["review_suggested"])

    def test_missing_or_revoked_permission_never_looks_ready(self):
        payload = card_payload()
        del payload["permission"]
        result = assess_json(payload)
        self.assertEqual(result["proposed_use"]["status"], "incomplete")
        self.assertEqual(result["proposed_use"]["reason_codes"], ["MISSING_PERMISSION_CARD"])
        self.assertTrue(result["incomplete"])

        payload = card_payload()
        event = copy.deepcopy(payload["permission"]["events"][0])
        event.update(sequence=2, action="revoke")
        payload["permission"]["events"].append(event)
        result = assess_json(payload)
        self.assertEqual(result["proposed_use"]["status"], "review")
        self.assertIn("P1:UNAUTHORIZED_REUSE", result["proposed_use"]["reason_codes"])

    def test_external_branch_requires_matching_gate_and_human_review_requirement(self):
        payload = card_payload()
        payload["proposed_use"]["external_action"] = True
        missing = assess_json(payload)["proposed_use"]
        self.assertEqual(missing["status"], "incomplete")
        self.assertEqual(missing["reason_codes"], ["MISSING_GATE_CARD"])
        payload["gate"] = {
            "review_owner": "person", "model_output_parsed": True,
            "accept_unparsed": False, "external_action_branch": False,
            "human_approval_required": True,
        }
        self.assertIn("GATE_BRANCH_MISMATCH", assess_json(payload)["proposed_use"]["reason_codes"])
        payload["gate"]["external_action_branch"] = True
        payload["gate"]["human_approval_required"] = False
        self.assertIn("G1:AUTO_EXTERNAL_ACTION", assess_json(payload)["proposed_use"]["reason_codes"])
        payload["gate"]["human_approval_required"] = True
        self.assertEqual(assess_json(payload)["proposed_use"]["status"], "ready_for_human_decision")
        payload["gate"]["model_output_parsed"] = False
        self.assertEqual(assess_json(payload)["proposed_use"]["status"], "incomplete")

    def test_review_and_missing_facts_are_both_visible(self):
        payload = card_payload()
        payload["proposed_use"]["external_action"] = True
        payload["routing"]["proposed_recipient"] = "unlisted"
        result = assess_json(payload)
        self.assertEqual(result["proposed_use"]["status"], "review")
        self.assertTrue(result["proposed_use"]["incomplete"])
        self.assertTrue(result["incomplete"])
        self.assertIn("MISSING_GATE_CARD", result["proposed_use"]["reason_codes"])
        self.assertIn("R1:PRINCIPAL_SCOPE_CONFLICT", result["proposed_use"]["reason_codes"])

    def test_gate_violation_does_not_hide_blocked_output(self):
        payload = card_payload()
        payload["proposed_use"]["external_action"] = True
        payload["gate"] = {
            "review_owner": "person", "model_output_parsed": False,
            "accept_unparsed": False, "external_action_branch": True,
            "human_approval_required": False,
        }
        result = assess_json(payload)
        self.assertEqual(result["proposed_use"]["status"], "review")
        self.assertTrue(result["proposed_use"]["incomplete"])
        self.assertIn("G1:OUTPUT_BLOCKED", result["proposed_use"]["reason_codes"])
        self.assertIn("G1:AUTO_EXTERNAL_ACTION", result["proposed_use"]["reason_codes"])

    def test_empty_proposal_and_bad_input(self):
        result = assess_json({"language": "uk", "proposed_use": {"external_action": False}})
        self.assertEqual(result["proposed_use"]["reason_codes"],
                         ["MISSING_ROUTING_CARD", "MISSING_PERMISSION_CARD"])
        with self.assertRaises(ValueError):
            assess_json({"language": "ro", "proposed_use": {"external_action": 1}})
        with self.assertRaises(ValueError):
            assess_proposed_use(permission=PermissionCard(
                "minor", "symbol", "A", "reader", 2, False,
                (PermissionEvent(1, "minor", "symbol", "A", "reader", "grant"),)))

    def test_local_branch_rejects_conflicting_external_gate(self):
        result = assess_proposed_use(
            routing=RoutingCard("minor", "symbol", "A", "reader", frozenset({"reader"}),
                                {("minor", "symbol", "A"): frozenset({"reader"})}),
            permission=PermissionCard("minor", "symbol", "A", "reader", 2, True,
                                      (PermissionEvent(1, "minor", "symbol", "A", "reader", "grant"),)),
            gate=GateCard("person", True, False, True, True))
        self.assertEqual(result.reason_codes, ("GATE_BRANCH_MISMATCH",))

    def test_jsonl_proposal_keeps_per_line_incomplete_state(self):
        import json
        lines = [card_payload(), {"language": "ro", "proposed_use": {"external_action": False}}]
        reports = assess_jsonl("\n".join(json.dumps(line) for line in lines))
        self.assertEqual([row["proposed_use"]["status"] for row in reports],
                         ["ready_for_human_decision", "incomplete"])

    def test_one_fact_contrast_reports_composed_use_change(self):
        before = card_payload()
        after = copy.deepcopy(before)
        after["proposed_use"]["external_action"] = True
        report = contrast_declared_contracts(before, after)
        self.assertEqual(report["changed_path"], "/proposed_use/external_action")
        self.assertEqual(report["transitions"], [])
        self.assertTrue(report["decision_changed"])
        self.assertEqual(report["proposed_use_transition"]["before_status"],
                         "ready_for_human_decision")
        self.assertEqual(report["proposed_use_transition"]["after_status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
