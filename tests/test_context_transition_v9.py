from collections import defaultdict

from training.context_transition_v9 import CONTENT, FRAMES, generate
from training.policy_entailment_v9 import targets


def test_current_attribution_and_irrelevant_history_are_separate():
    for language in ('ro', 'uk'):
        for partition, count in (('train', 864), ('dev', 144)):
            rows = generate(language, partition)
            assert len(rows) == count
            assert len({r['case_id'] for r in rows}) == count
            invariance, pairs = defaultdict(list), defaultdict(list)
            for row in rows:
                assert int(any(targets(row))) == row['review']
                assert row['synthetic_only'] and not row['independently_annotated']
                assert row['focus'] == 1
                invariance[row['invariance_group']].append(row)
                pairs[row['pair_group']].append(row)
            for group in invariance.values():
                assert len(group) == 3
                assert len({r['turns'][0]['text'] for r in group}) == 3
                assert len({r['turns'][1]['text'] for r in group}) == 1
                assert len({tuple(targets(r)) for r in group}) == 1
            for group in pairs.values():
                assert len(group) == 2
                if group[0]['attributes']['concern'] == '1':
                    assert sorted(r['review'] for r in group) == [0, 1]
                else:
                    assert [r['review'] for r in group] == [0, 0]


def test_development_has_disjoint_bodies_framing_and_conversations():
    for language in ('ro', 'uk'):
        bodies = {p: {s for rows in CONTENT[language][p].values() for s in rows} for p in ('train', 'dev')}
        frames = {p: {s for pair in FRAMES[language][p] for s in pair} for p in ('train', 'dev')}
        assert not bodies['train'] & bodies['dev']
        assert not frames['train'] & frames['dev']
        messages = {p: {r['turns'][1]['text'] for r in generate(language, p)} for p in ('train', 'dev')}
        assert not messages['train'] & messages['dev']
