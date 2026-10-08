"""Corrected v1 one-fact exploration stays bounded and value-free."""

import unittest

from roguard.v1_explore import explore_v1_contract


class V1ExploreTests(unittest.TestCase):
    def test_detects_independent_disclosure_and_route_changes(self):
        payload = {"language": "ro",
                   "disclosure": {"source_role": "minor", "safety_or_support_anchor": False,
                                  "indirect_or_repeated_support_pattern": False,
                                  "explicit_support_request": False,
                                  "direct_safety_statement": True, "retracted_or_denied": True},
                   "routing": {"principal": "minor", "item": "symbol", "purpose": "A",
                               "proposed_recipient": "guardian", "recipient_roles": ["minor", "guardian"],
                               "allowed_by_scope": [{"principal": "minor", "item": "symbol",
                                                     "purpose": "A", "recipients": ["guardian"]}],
                               "recipient_may_be_source_of_concern": "unknown"}}
        result = explore_v1_contract(payload)
        paths = {change["path"] for change in result["decision_changes"]}
        self.assertIn("/disclosure/direct_safety_statement", paths)
        self.assertIn("/routing/recipient_may_be_source_of_concern", paths)
        self.assertNotIn("/disclosure/retracted_or_denied", paths)
        self.assertFalse(result["external_action_taken"])
        self.assertNotIn("symbol", str(result))
        self.assertTrue(all(change["target_state_code"] for change in result["decision_changes"]))

    def test_support_changes_and_limit_are_explicit(self):
        payload = {"language": "uk", "support": {
            "applicable_fields": ["next_step", "no_pressing_for_details"],
            "passed_fields": ["next_step"]}}
        full = explore_v1_contract(payload)
        self.assertEqual({change["path"] for change in full["decision_changes"]},
                         {"/support/passed_fields/next_step",
                          "/support/passed_fields/no_pressing_for_details"})
        limited = explore_v1_contract(payload, limit=1)
        self.assertTrue(limited["truncated"])
        self.assertEqual(limited["tested"], 1)

    def test_invalid_card_and_limit_fail_closed(self):
        with self.assertRaises(ValueError):
            explore_v1_contract({"language": "ro"})
        with self.assertRaises(ValueError):
            explore_v1_contract({"language": "ro", "support": {
                "applicable_fields": [], "passed_fields": []}}, limit=0)


if __name__ == "__main__":
    unittest.main()
