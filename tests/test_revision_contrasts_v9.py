from collections import Counter, defaultdict

from roguard.evidence import disposition
from training.policy_entailment_v9 import targets
from training.revision_contrasts_v9 import generate


def test_revision_pairs_preserve_policy_for_both_languages_and_scopes():
    for language in ('ro', 'uk'):
        rows = generate(language)
        assert len(rows) == 1152
        from training.train_paired_nli_v9 import premise
        assert len({premise(r) for r in rows}) == len(rows)
        groups = defaultdict(list)
        for r in rows:
            decision = disposition({k: {'label': v} for k, v in r['attributes'].items()})[0]
            assert int(decision != 'CONTINUE') == r['review']
            assert targets(r) == (0, r['review'])
            assert r['partition'] == 'train'
            assert 0 <= r['focus'] < len(r['turns'])
            groups[r['pair_group']].append(r)
        assert all(sorted(r['review'] for r in pair) == [0, 1] for pair in groups.values())
        assert Counter(r['scope'] for r in rows) == {'self': 384, 'reported': 384, 'fiction': 384}


def test_generated_content_has_no_exact_overlap_with_frozen_partitions():
    from training.paired_scope_data_v9 import generate as scope
    from training.train_paired_nli_v9 import premise
    for language in ('ro', 'uk'):
        protected = {premise(r) for split in ('dev', 'test') for r in scope(language, split)}
        assert not protected & {premise(r) for r in generate(language)}
