import pytest


@pytest.mark.parametrize('language', ['ro', 'uk'])
def test_scope_repairs_include_contrastive_denial_and_contextual_fiction(language):
    from roguard.evidence import ONTOLOGY
    from roguard.native_review import prepare_input
    from training.scope_repair_data_v9 import generate
    rows = generate(language)
    assert len(rows) == 5184
    identities = set()
    for row in rows:
        assert row['synthetic'] is True and row['focus'] == 2
        attrs = row['attributes']
        assert set(attrs) == set(ONTOLOGY)
        assert all(attrs[f] in ONTOLOGY[f] for f in ONTOLOGY)
        assert (attrs['subject'] == 'FICTIONAL') == (attrs['source'] == 'FICTION')
        assert (attrs['assertion'] == 'RETRACTED') == (attrs['relation'] == 'RETRACTS')
        identity = prepare_input(dict(language=language, turns=row['turns']), 2)[0]
        assert identity not in identities
        identities.add(identity)
    assert {r['attributes']['assertion'] for r in rows} == set(ONTOLOGY['assertion'])
    # Both affirmative denials-of-denial and actual denials occur with both
    # concern labels, so the presence of a negation token cannot solve the task.
    for label in ('AFFIRMED', 'NEGATED'):
        assert {r['attributes']['concern'] for r in rows if r['attributes']['assertion'] == label} == {'0', '1'}
