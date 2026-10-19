from collections import Counter, defaultdict

from training.scope_operators_v9 import generate
from training.paired_scope_data_v9 import CONTENT, BENIGN
from training.prepare_instruction_sft_v8 import target


def test_scope_operators_have_consistent_policy_labels_and_complete_opposed_pairs():
    for language in ('ro', 'uk'):
        rows = generate(language)
        assert len(rows) == 1920
        assert Counter(r['scope'] for r in rows) == dict(denial=864, fiction=864, reset=192)
        assert len({r['case_id'] for r in rows}) == len(rows)
        groups = defaultdict(list)
        for row in rows:
            assert row['review'] == target(row)
            assert row['partition'] == 'train' and row['synthetic_only'] is True
            assert row['independently_annotated'] is False
            assert row['focus'] == len(row['turns'])-1
            if row['pair_group']:
                groups[row['pair_group']].append(row)
        assert len(groups) == 672
        for pair in groups.values():
            assert sorted(r['review'] for r in pair) == [0, 1]
            if pair[0]['scope'] == 'reset':
                assert pair[0]['turns'][0] == pair[1]['turns'][0]


def test_ordinary_requests_prevent_scope_wording_from_becoming_a_positive_shortcut():
    for language in ('ro', 'uk'):
        ordinary = [r for r in generate(language) if r['pair_group'] is None]
        assert len(ordinary) == 576
        assert all(r['review'] == 0 and r['attributes']['concern'] == '0' for r in ordinary)
        assert {r['attributes']['source'] for r in ordinary} == {'FICTION', 'DIRECT'}
        assert {r['attributes']['assertion'] for r in ordinary} == {'NEGATED', 'AFFIRMED'}


def test_scope_operator_examples_do_not_copy_protected_content():
    for language in ('ro', 'uk'):
        protected = [x for partition in ('dev', 'test')
                     for x in CONTENT[language][partition]+BENIGN[language][partition]]
        rendered = [' '.join(t['text'] for t in r['turns']) for r in generate(language)]
        assert all(not any(x in text for x in protected) for text in rendered)
