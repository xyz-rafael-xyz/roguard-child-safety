import unittest
from training.audit_review_threshold_v8 import feasibility


class ThresholdAuditTests(unittest.TestCase):
    def test_tied_negative_cannot_be_removed_without_losing_positive(self):
        result = feasibility({'one': [{'gold': 1, 'score': .5}, {'gold': 0, 'score': .5}]})
        self.assertFalse(result['single_threshold_can_meet_targets_on_inspected_rows'])
        self.assertEqual(result['metrics']['one']['missed'], 0)

    def test_one_percent_boundary_and_multiple_groups(self):
        groups = {'a': [{'gold': 1, 'score': .1}]+[{'gold': 1, 'score': .8}]*99+[{'gold': 0, 'score': .2}]*20,
                  'b': [{'gold': 1, 'score': .6}, {'gold': 0, 'score': .3}]}
        result = feasibility(groups)
        self.assertEqual(result['largest_threshold_preserving_99_percent_recall_in_every_group'], .6)
        self.assertEqual(result['metrics']['a']['missed'], 1)
        self.assertTrue(result['single_threshold_can_meet_targets_on_inspected_rows'])

    def test_invalid_data(self):
        for groups in ({}, {'a': []}, {'a': [{'gold': 1, 'score': .2}]},
                       {'a': [{'gold': 1, 'score': float('nan')}, {'gold': 0, 'score': .1}]}):
            with self.assertRaises(ValueError):
                feasibility(groups)
