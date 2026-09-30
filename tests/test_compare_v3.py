import unittest

from eval.compare_v3 import summarize


class V3ComparisonTests(unittest.TestCase):
    def test_pair_metrics_expose_always_positive_shortcut(self):
        rows = [
            {"id": "a0", "language": "ro", "source_kind": "message", "labels": []},
            {"id": "a1", "language": "ro", "source_kind": "message", "labels": ["D1"]},
            {"id": "b0", "language": "ro", "source_kind": "routing_card", "labels": []},
            {"id": "b1", "language": "ro", "source_kind": "routing_card", "labels": ["R1"]},
        ]
        output = {"predictions": [
            {"id": "a0", "predicted": ["D1"], "strict_parse": True},
            {"id": "a1", "predicted": ["D1"], "strict_parse": True},
            {"id": "b0", "predicted": [], "strict_parse": True},
            {"id": "b1", "predicted": ["R1"], "strict_parse": True},
        ]}
        report = summarize(rows, output, "ro-v3-adapter")
        self.assertEqual(report["exact_rows"], 3)
        self.assertEqual(report["exact_pairs"], 1)
        self.assertEqual(report["correct_direction_flips"], 1)
        self.assertEqual(report["false_review_on_negatives"], 1)


if __name__ == "__main__":
    unittest.main()
