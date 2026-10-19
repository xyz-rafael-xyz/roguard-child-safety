from collections import Counter, defaultdict

from training.indirect_distress_v9 import generate, CONTENT, ORDINARY
from training.paired_scope_data_v9 import CONTENT as PREVIOUS_CONTENT, BENIGN
from training.prepare_instruction_sft_v8 import target


def test_indirect_content_has_consistent_labels_complete_pairs_and_same_reset_history():
    for language in ('ro', 'uk'):
        rows = generate(language)
        assert len(rows) == 1272
        assert Counter(r['scope'] for r in rows) == dict(bare=24, denial=576, fiction=576, reset=96)
        assert len({r['case_id'] for r in rows}) == len(rows)
        assert Counter(r['review'] for r in rows) == {0: 924, 1: 348}
        pairs = defaultdict(list)
        for row in rows:
            assert row['review'] == target(row)
            assert row['partition'] == 'train' and row['synthetic_only']
            assert not row['independently_annotated']
            assert row['focus'] == len(row['turns']) - 1
            if row['pair_group']:
                pairs[row['pair_group']].append(row)
        assert len(pairs) == 336
        for pair in pairs.values():
            assert sorted(r['review'] for r in pair) == [0, 1]
            if pair[0]['scope'] == 'reset':
                assert pair[0]['turns'][0] == pair[1]['turns'][0]


def test_ordinary_bodily_vocabulary_has_bare_and_both_scope_controls():
    for language in ('ro', 'uk'):
        rows = generate(language)
        ordinary = [r for r in rows if r['pair_group'] is None and r['review'] == 0]
        assert len(ordinary) == 588
        for content in ORDINARY[language]:
            matching = [r for r in ordinary if any(content in t['text'] for t in r['turns'])]
            assert len(matching) == 49
            assert {r['scope'] for r in matching} == {'bare', 'denial', 'fiction'}
        bare_positive = [r for r in rows if r['scope'] == 'bare' and r['review']]
        assert [r['turns'][0]['text'] for r in bare_positive] == CONTENT[language]


def test_new_content_does_not_copy_any_previous_scope_content_partition():
    for language in ('ro', 'uk'):
        previous = [x for partition in ('train', 'dev', 'test')
                    for x in PREVIOUS_CONTENT[language][partition]+BENIGN[language][partition]]
        rendered = [' '.join(t['text'] for t in r['turns']) for r in generate(language)]
        assert all(not any(x in text for x in previous) for text in rendered)
