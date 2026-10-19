from collections import Counter, defaultdict

from training.freeform_scope_v9 import generate, PAIRS
from training.prepare_instruction_sft_v8 import target


def test_freeform_policy_labels_complete_pairs_and_preserved_context():
    for language in ('ro', 'uk'):
        rows = generate(language)
        assert len(rows) == 60
        assert len({r['case_id'] for r in rows}) == len(rows)
        assert Counter(r['review'] for r in rows) == {0:36, 1:24}
        pairs = defaultdict(list)
        for row in rows:
            assert row['review'] == target(row)
            assert row['partition'] == 'train' and row['synthetic_only']
            assert not row['independently_annotated']
            assert row['focus'] == len(row['turns'])-1
            if row['pair_group']:
                pairs[row['pair_group']].append(row)
        assert len(pairs) == 24
        for pair in pairs.values():
            assert sorted(r['review'] for r in pair) == [0, 1]
            if pair[0]['scope'] == 'reset':
                assert pair[0]['turns'][0] == pair[1]['turns'][0]
        assert Counter(r['scope'] for r in rows)['reset'] == 8


def test_freeform_rows_use_authored_text_without_unexpanded_templates():
    for language in ('ro','uk'):
        texts = [tuple(t['text'] for t in r['turns']) for r in generate(language)]
        assert len(set(texts)) == len(texts)
        assert all('{' not in text and '}' not in text for turns in texts for text in turns)
        assert len(PAIRS[language]) == 24
