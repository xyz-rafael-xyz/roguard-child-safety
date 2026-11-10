from copy import deepcopy
import unittest

from roguard.assurance import digest
from training.matched_role_diversity_v10 import matched_population


def row(name, token, *, gold=0, partition='train', new=False):
    return dict(case_id=name, language='ro', partition=partition,
                fields=[0, 0, 0, 0, 0, gold], gold=gold,
                tokens=[token], token_count=1, token_sha256=digest([token]),
                synthetic_only=new, independently_annotated=False)


class MatchedRolePopulationTests(unittest.TestCase):
    def setUp(self):
        self.base = [row('dev-positive', 1, gold=1, partition='dev'),
                     row('negative-a', 2), row('positive-a', 3, gold=1),
                     row('negative-b', 4), row('dev-negative', 5, partition='dev')]
        self.additions = [row('new-'+str(i), 10+i, gold=i % 2, new=True) for i in range(12)]

    def test_matching_preserves_development_and_per_slot_supervision(self):
        original = deepcopy((self.base, self.additions))
        result = matched_population(self.base, self.additions, 'ro')
        self.assertEqual((self.base, self.additions), original)
        for arm in ('control', 'treatment'):
            self.assertEqual(result[arm][:len(self.base)], self.base)
            self.assertEqual([r for r in result[arm] if r['partition'] == 'dev'],
                             [self.base[0], self.base[4]])
        for a, b in zip(result['control'], result['treatment']):
            self.assertEqual((a['fields'], a['gold'], a['partition']),
                             (b['fields'], b['gold'], b['partition']))
        self.assertEqual(result['summary']['train_rows_per_arm'], 15)
        self.assertEqual(result['summary']['control_distinct_repeated_inputs'], 3)
        self.assertEqual(result['summary']['control_max_additional_repetitions'], 6)
        counts = {name: sum(a['control_case_id'] == name for a in result['assignments'])
                  for name in ('negative-a', 'negative-b', 'positive-a')}
        self.assertEqual(counts, {'negative-a': 3, 'negative-b': 3, 'positive-a': 6})
        self.assertEqual(result, matched_population(deepcopy(self.base), deepcopy(self.additions), 'ro'))

    def test_development_cannot_supply_missing_control_cell(self):
        self.base = [r for r in self.base if r['case_id'] != 'positive-a']
        with self.assertRaisesRegex(ValueError, 'no original TRAIN'):
            matched_population(self.base, self.additions, 'ro')

    def test_wrong_partition_provenance_and_language_rejected(self):
        for key, value in [('partition', 'dev'), ('partition', 'test'), ('language', 'uk'),
                           ('synthetic_only', False), ('independently_annotated', True)]:
            with self.subTest(key=key, value=value):
                additions = deepcopy(self.additions); additions[0][key] = value
                with self.assertRaises(ValueError):
                    matched_population(self.base, additions, 'ro')

    def test_development_identity_and_token_overlap_rejected(self):
        for kind in ('id', 'tokens'):
            with self.subTest(kind=kind):
                additions = deepcopy(self.additions)
                if kind == 'id':
                    additions[0]['case_id'] = self.base[0]['case_id']
                else:
                    for key in ('tokens', 'token_count', 'token_sha256'):
                        additions[0][key] = self.base[0][key]
                with self.assertRaisesRegex(ValueError, 'Repeated source'):
                    matched_population(self.base, additions, 'ro')

    def test_corrupt_labels_and_tokens_rejected(self):
        for key, value in [('gold', True), ('gold', 2), ('fields', [0]*5),
                           ('fields', [True, 0, 0, 0, 0, 0]), ('fields', [9, 0, 0, 0, 0, 0]),
                           ('tokens', [999]), ('tokens', [True]), ('token_count', 2)]:
            with self.subTest(key=key, value=value):
                additions = deepcopy(self.additions); additions[0][key] = value
                with self.assertRaises(ValueError):
                    matched_population(self.base, additions, 'ro')

    def test_control_reference_positions_resolve_to_original_train_only(self):
        result = matched_population(self.base, self.additions, 'ro')
        for ref, assignment in zip(result['references']['control'][len(self.base):], result['assignments']):
            self.assertEqual(ref[0], 'base')
            source = self.base[ref[1]]
            self.assertEqual(source['partition'], 'train')
            self.assertEqual(source['case_id'], assignment['control_case_id'])
        self.assertEqual(result['references']['treatment'][len(self.base):],
                         [('new', i) for i in range(len(self.additions))])


if __name__ == '__main__':
    unittest.main()
