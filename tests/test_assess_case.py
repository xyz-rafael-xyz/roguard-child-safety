import unittest

from roguard import (BoundaryCard, ReadabilityCard, RoutingCard, assess_case)
from roguard.review import CATEGORIES


class FakeBackend:
    def __init__(self, positive: str):
        self.positive = positive
        self.calls = 0

    def score(self, text: str, language: str, source_kind: str) -> dict[str, float]:
        self.calls += 1
        return {code: float(code == self.positive) for code in CATEGORIES}


THRESHOLDS = {code: 0.5 for code in CATEGORIES}


class AssessCaseTests(unittest.TestCase):
    def test_complete_policy_takes_precedence_in_one_call(self):
        backend = FakeBackend("R1")
        route = RoutingCard("minor", "item", "A", "minor", frozenset({"minor", "guardian"}),
                            {("minor", "item", "A"): frozenset({"minor"})})
        report = assess_case(language="ro", text="Fișă simbolică neutră.",
                             source_kind="routing_card", backend=backend,
                             thresholds=THRESHOLDS, routing=route)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(report.policy_disagreements, ("R1",))
        self.assertFalse(report.review_suggested)

    def test_contract_validation_runs_before_backend(self):
        backend = FakeBackend("P1")
        invalid = BoundaryCard(frozenset({"x"}), frozenset({"x"}), frozenset(),
                               frozenset(), False, False)
        with self.assertRaisesRegex(ValueError, "Contradictory boundary"):
            assess_case(language="ro", text="Fișă simbolică neutră.",
                        source_kind="boundary_card", backend=backend,
                        thresholds=THRESHOLDS, boundary=invalid)
        self.assertEqual(backend.calls, 0)

    def test_ukrainian_contract_path_does_not_need_a_model(self):
        report = assess_case(language="uk", readability=ReadabilityCard(10, 5, 4))
        self.assertEqual(report.contracts.findings[0].reason_codes, ("WORD_CAP_EXCEEDED",))
        self.assertTrue(report.review_suggested)
        backend = FakeBackend("A1")
        with self.assertRaisesRegex(ValueError, "active taxonomy language"):
            assess_case(language="uk", text="Символічна картка.",
                        source_kind="response", backend=backend,
                        thresholds=THRESHOLDS, readability=ReadabilityCard(10, 3, 4))
        self.assertEqual(backend.calls, 0)

    def test_text_configuration_is_all_or_none(self):
        with self.assertRaisesRegex(ValueError, "Supply declared contracts"):
            assess_case(language="ro")
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            assess_case(language="ro", text="Fișă neutră")
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            assess_case(language="ro", backend=FakeBackend("D1"),
                        readability=ReadabilityCard(10, 3, 4))


if __name__ == "__main__":
    unittest.main()
