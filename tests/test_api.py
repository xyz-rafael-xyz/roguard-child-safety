import unittest

from roguard import CATEGORIES, MLXClassifier, calibrate, escalate, screen
from roguard.prompt import make_prompt, make_prompt_v2


def _calibration_rows():
    kinds = {"D1": "message", "R1": "routing_card", "A1": "response",
             "P1": "permission_card", "G1": "gate_card", "S1": "response"}
    rows = []
    for code, kind in kinds.items():
        rows.append(({name: 0.9 if name == code else 0.1 for name in CATEGORIES},
                     {code}, kind))
        rows.append(({name: 0.2 for name in CATEGORIES}, set(), kind))
    return rows


class Backend:
    def score(self, text, language, source_kind):
        return {code: (0.8 if code == "D1" else 0.1) for code in CATEGORIES}


class ApiTests(unittest.TestCase):
    def test_screen_and_return_only_signal(self):
        thresholds = {code: 0.5 for code in CATEGORIES}
        result = screen("Abstract card", language="ro", source_kind="message", backend=Backend(), thresholds=thresholds)
        self.assertEqual(result.labels, ("D1",))
        signal = escalate(result)
        self.assertTrue(signal.review_suggested)
        self.assertEqual(signal.labels, ("D1",))
        self.assertEqual(signal.reason_codes, ("REVIEW_SUPPORT_SIGNAL",))

    def test_non_message_cannot_receive_disclosure_signal(self):
        thresholds = {code: 0.5 for code in CATEGORIES}
        result = screen("Abstract card", language="ro", source_kind="response", backend=Backend(), thresholds=thresholds)
        self.assertEqual(result.labels, ())

    def test_screen_rejects_nonfinite_scores_and_thresholds(self):
        class NonfiniteBackend:
            def score(self, text, language, source_kind):
                return {code: (float("nan") if code == "D1" else 0.1) for code in CATEGORIES}

        thresholds = {code: 0.5 for code in CATEGORIES}
        with self.assertRaisesRegex(ValueError, "Invalid score"):
            screen("Abstract card", language="ro", source_kind="message",
                   backend=NonfiniteBackend(), thresholds=thresholds)
        thresholds["D1"] = float("inf")
        with self.assertRaisesRegex(ValueError, "Invalid score"):
            screen("Abstract card", language="ro", source_kind="message",
                   backend=Backend(), thresholds=thresholds)
        thresholds["D1"] = True
        with self.assertRaisesRegex(ValueError, "Invalid score"):
            screen("Abstract card", language="ro", source_kind="message",
                   backend=Backend(), thresholds=thresholds)

    def test_screen_checks_inputs_before_calling_backend(self):
        class CountingBackend:
            calls = 0

            def score(self, text, language, source_kind):
                self.calls += 1
                return {code: 0.5 for code in CATEGORIES}

        backend = CountingBackend()
        thresholds = {code: 0.5 for code in CATEGORIES}
        thresholds["D1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid score"):
            screen("Abstract card", language="ro", source_kind="message",
                   backend=backend, thresholds=thresholds)
        with self.assertRaisesRegex(ValueError, "nonempty text"):
            screen(None, language="ro", source_kind="message",
                   backend=backend, thresholds={code: 0.5 for code in CATEGORIES})
        with self.assertRaisesRegex(ValueError, "threshold"):
            screen("Abstract card", language="ro", source_kind="message",
                   backend=backend, thresholds={"D1": 0.5})
        self.assertEqual(backend.calls, 0)

    def test_boundary_card_routes_persistence_signal_only(self):
        class BoundaryBackend:
            def score(self, text, language, source_kind):
                return {code: (0.8 if code in {"D1", "P1"} else 0.1) for code in CATEGORIES}

        thresholds = {code: 0.5 for code in CATEGORIES}
        result = screen("Abstract boundary card", language="ro", source_kind="boundary_card", backend=BoundaryBackend(), thresholds=thresholds)
        self.assertEqual(result.labels, ("P1",))
        self.assertEqual(escalate(result).reason_codes, ("REVIEW_PERSISTENCE",))

    def test_calibration_requires_positive_for_each_code(self):
        with self.assertRaisesRegex(ValueError, "No positive"):
            calibrate([({code: 0.2 for code in CATEGORIES}, set(), "message")])

    def test_calibration_requires_both_classes_and_valid_card_labels(self):
        rows = _calibration_rows()
        self.assertEqual(calibrate(rows), {code: 0.9 for code in CATEGORIES})
        with self.assertRaisesRegex(ValueError, "No negative development case for D1"):
            calibrate([row for row in rows if row[2] != "message" or row[1]])
        incompatible = list(rows)
        scores, _, kind = incompatible[0]
        incompatible[0] = (scores, {"S1"}, kind)
        with self.assertRaisesRegex(ValueError, "inapplicable labels"):
            calibrate(incompatible)
        unknown = list(rows)
        scores, _, kind = unknown[0]
        unknown[0] = (scores, {"UNKNOWN"}, kind)
        with self.assertRaisesRegex(ValueError, "invalid or inapplicable labels"):
            calibrate(unknown)

    def test_calibration_rejects_malformed_scores_and_target(self):
        rows = _calibration_rows()
        for target in (True, float("nan"), float("inf"), 0):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "target_recall"):
                calibrate(rows, target_recall=target)
        for bad in (True, float("nan"), float("inf"), -0.1, 1.1):
            altered = list(rows)
            scores, labels, kind = altered[0]
            altered[0] = ({**scores, "D1": bad}, labels, kind)
            with self.subTest(score=bad), self.assertRaisesRegex(ValueError, "invalid probabilities"):
                calibrate(altered)
        altered = list(rows)
        scores, labels, kind = altered[0]
        altered[0] = ({name: value for name, value in scores.items() if name != "S1"},
                      labels, kind)
        with self.assertRaisesRegex(ValueError, "must score every category"):
            calibrate(altered)

    def test_mlx_hard_codes_are_strict_and_kind_scoped(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]

        backend = object.__new__(MLXClassifier)
        backend.language = "ro"
        backend.prompt_fn = make_prompt
        backend.tokenizer = Tokenizer()
        backend.model = object()
        backend.generate = lambda *args, **kwargs: "D1"
        thresholds = {code: 0.5 for code in CATEGORIES}
        result = screen("Abstract card", language="ro", source_kind="message",
                        backend=backend, thresholds=thresholds)
        self.assertEqual(result.labels, ("D1",))
        with self.assertRaisesRegex(ValueError, "parsed"):
            screen("Abstract card", language="ro", source_kind="response",
                   backend=backend, thresholds=thresholds)

        backend.prompt_fn = make_prompt_v2
        result = screen("Abstract card", language="ro", source_kind="message",
                        backend=backend, thresholds=thresholds)
        self.assertEqual(result.labels, ("D1",))

    def test_binary_adapter_checks_response_categories_separately(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]

        backend = object.__new__(MLXClassifier)
        backend.language = "ro"
        backend.prompt_version = "v4"
        backend.tokenizer = Tokenizer()
        backend.model = object()
        backend.generate = lambda *args, **kwargs: "da" if "Regula pentru S1" in kwargs["prompt"] else "nu"
        result = screen("Fișă simbolică", language="ro", source_kind="response", backend=backend,
                        thresholds={code: 0.5 for code in CATEGORIES})
        self.assertEqual(result.labels, ("S1",))
        backend.generate = lambda *args, **kwargs: "da, poate"
        with self.assertRaisesRegex(ValueError, "parsed"):
            backend.score("Fișă simbolică", "ro", "response")


if __name__ == "__main__":
    unittest.main()
