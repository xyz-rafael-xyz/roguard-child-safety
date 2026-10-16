import unittest

from training.analyze_instruction_errors_v8 import analyze
from training.context_families_v8 import generate
from training.prepare_instruction_sft_v8 import target


class InstructionErrorSlicesTests(unittest.TestCase):
    def test_unresolved_counts_as_error_for_both_classes(self):
        rows = generate()['dev']
        selected = [next(row for row in rows if target(row) == gold) for gold in (0, 1)]
        predictions = [dict(case_id=row['case_id'], language=row['language'], predicted=None)
                       for row in selected]
        report = analyze(selected, predictions)
        groups = [counts for name, counts in report['groups'].items() if '/input_source/' in name]
        self.assertEqual(sum(group['missed'] for group in groups), 1)
        self.assertEqual(sum(group['false_reviews'] for group in groups), 1)
        self.assertEqual(sum(group['unresolved'] for group in groups), 2)

    def test_missing_predictions_cannot_hide_errors(self):
        with self.assertRaises(ValueError):
            analyze(generate()['dev'][:2], [])
