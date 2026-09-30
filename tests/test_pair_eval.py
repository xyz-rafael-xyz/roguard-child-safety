import unittest

from roguard.pair_eval import evaluate_pairs


class PairEvalTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "n", "language": "ro", "source_kind": "routing_card", "labels": []},
            {"id": "p", "language": "ro", "source_kind": "routing_card", "labels": ["R1"]},
        ]

    def test_pair_metrics_expose_false_reviews(self):
        correct = [{"id": "n", "predicted": [], "strict_parse": True},
                   {"id": "p", "predicted": ["R1"], "strict_parse": True}]
        result = evaluate_pairs(self.rows, correct)
        self.assertEqual((result["exact_cards"], result["exact_pairs"],
                          result["correct_direction_flips"]), (2, 1, 1))
        over_review = [{"id": "n", "predicted": ["R1"], "strict_parse": True}, correct[1]]
        result = evaluate_pairs(self.rows, over_review)
        self.assertEqual((result["exact_cards"], result["exact_pairs"],
                          result["false_review_on_negatives"]), (1, 0, 1))

    def test_tampered_labels_and_inapplicable_predictions_fail(self):
        with self.assertRaisesRegex(ValueError, "expected label"):
            evaluate_pairs(self.rows, [{"id": "n", "expected": ["R1"], "predicted": [], "strict_parse": True},
                                       {"id": "p", "predicted": ["R1"], "strict_parse": True}])
        with self.assertRaisesRegex(ValueError, "applicable"):
            evaluate_pairs(self.rows, [{"id": "n", "predicted": ["D1"], "strict_parse": True},
                                       {"id": "p", "predicted": ["R1"], "strict_parse": True}])


if __name__ == "__main__":
    unittest.main()
