from collections import Counter, defaultdict
import json
import random

from training.focus_grounding_v9 import generate
from training.shared_tail_training_v9 import plan_batches


def test_grounding_families_preserve_focus_but_keep_retractions_sensitive_to_history():
    for language in ('ro', 'uk'):
        all_keys = set()
        for partition in ('train', 'dev'):
            rows = generate(language, partition)
            assert len(rows) == (1728 if partition == 'train' else 120)
            assert set(Counter(r['review'] for r in rows)) == {0, 1}
            groups = defaultdict(list)
            for row in rows:
                key = json.dumps(row['turns'], ensure_ascii=False, sort_keys=True)
                assert key not in all_keys
                all_keys.add(key)
                assert row['synthetic_only'] and not row['independently_annotated']
                groups[row['context_family']].append(row)
            assert all(len(v) == 6 for v in groups.values())
            for group in groups.values():
                if '-query-' in group[0]['context_family']:
                    for label in (0, 1):
                        matched = [r for r in group if r['review'] == label]
                        assert len(matched) == 3
                        assert len({r['turns'][r['focus']]['text'] for r in matched}) == 1
                else:
                    denials = [r for r in group if r['attributes']['assertion'] in ('NEGATED', 'RETRACTED')]
                    assert sorted(r['review'] for r in denials) == [0, 0, 1]
                    assert len({r['turns'][r['focus']]['text'] for r in denials}) == 1
                    assert next(r for r in denials if r['review'])['attributes']['relation'] == 'RETRACTS'
            batches = plan_batches(rows, random.Random(1173), 8)
            assert sorted(i for indices, _, _ in batches for i in indices) == list(range(len(rows)))
