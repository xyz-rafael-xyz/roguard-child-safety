"""Independent state-machine audit of permission event histories."""

import itertools
import unittest

from roguard import PermissionCard, PermissionEvent, check_permission


def expected_active(actions: tuple[str, ...]) -> bool:
    """Grant starts a lease; pause preserves it; revoke ends it."""
    state = "absent"
    for action in actions:
        if action == "grant":
            state = "active"
        elif action in {"narrow", "revoke"}:
            state = "absent"
        elif action == "pause" and state == "active":
            state = "paused"
        elif action == "resume" and state == "paused":
            state = "active"
    return state == "active"


class PermissionStateMatrixTests(unittest.TestCase):
    def test_every_history_up_to_five_events(self):
        actions = ("grant", "narrow", "revoke", "pause", "resume")
        for length in range(1, 6):
            for history in itertools.product(actions, repeat=length):
                events = tuple(PermissionEvent(index, "minor", "K", "A", "recipient", action)
                               for index, action in enumerate(history, 1))
                for proposed_use, expected_reason in (
                    (True, () if expected_active(history) else ("UNAUTHORIZED_REUSE",)),
                    (False, ("OVER_WITHHOLD",) if expected_active(history) else ()),
                ):
                    card = PermissionCard("minor", "K", "A", "recipient", length + 1,
                                          proposed_use, events)
                    self.assertEqual(check_permission(card).reason_codes, expected_reason,
                                     (history, proposed_use))

    def test_scope_and_expiry_cannot_be_bypassed(self):
        grant = PermissionEvent(1, "minor", "K", "A", "recipient", "grant", expires_at=4)
        foreign = PermissionEvent(1, "other", "K", "A", "recipient", "grant")
        card = PermissionCard("minor", "K", "A", "recipient", 3, True, (grant, foreign))
        self.assertEqual(check_permission(card).reason_codes, ())
        expired = PermissionCard("minor", "K", "A", "recipient", 4, True, (grant, foreign))
        self.assertEqual(check_permission(expired).reason_codes, ("UNAUTHORIZED_REUSE",))
        renewed = PermissionEvent(3, "minor", "K", "A", "recipient", "grant")
        current = PermissionCard("minor", "K", "A", "recipient", 4, True, (grant, foreign, renewed))
        self.assertEqual(check_permission(current).reason_codes, ())
        duplicate = PermissionEvent(1, "minor", "K", "A", "recipient", "revoke")
        ambiguous = PermissionCard("minor", "K", "A", "recipient", 3, True, (grant, duplicate))
        with self.assertRaisesRegex(ValueError, "unique within a scope"):
            check_permission(ambiguous)

    def test_narrowing_ends_only_named_scope_until_new_grant(self):
        events = (
            PermissionEvent(1, "minor", "K", "A", "recipient", "grant"),
            PermissionEvent(1, "minor", "K", "B", "recipient", "grant"),
            PermissionEvent(2, "minor", "K", "A", "recipient", "narrow"),
            PermissionEvent(3, "minor", "K", "A", "recipient", "resume"),
        )
        narrowed = PermissionCard("minor", "K", "A", "recipient", 4, True, events)
        retained = PermissionCard("minor", "K", "B", "recipient", 4, True, events)
        self.assertEqual(check_permission(narrowed).reason_codes, ("UNAUTHORIZED_REUSE",))
        self.assertEqual(check_permission(retained).reason_codes, ())
        renewed = events + (PermissionEvent(5, "minor", "K", "A", "recipient", "grant"),)
        self.assertEqual(check_permission(PermissionCard(
            "minor", "K", "A", "recipient", 6, True, renewed)).reason_codes, ())


if __name__ == "__main__":
    unittest.main()
