from copy import deepcopy

import pytest
from roguard.assurance import assure,answer,digest
from roguard.evidence import ONTOLOGY
from roguard.review_labels import PROMPTS,OPTIONS,question_view


@pytest.mark.parametrize('language',['ro','uk'])
def test_localized_view_keeps_exact_branches_and_binding(language):
    a = {field:labels[0] for field,labels in ONTOLOGY.items()}
    b = dict(a,concern='1')
    certificate = assure(dict(language=language,hypotheses=[a,b],context_digest=digest('fictional fixture')))
    before = deepcopy(certificate); view = question_view(certificate)
    assert certificate == before
    assert view['field'] == 'concern'
    assert {option['value'] for option in view['options']} == set(certificate['tree']['branches'])
    assert view['certificate_digest'] == certificate['certificate_digest']
    resolved = answer(certificate,{view['field']:view['options'][0]['value']},expected_certificate_digest=view['certificate_digest'])
    assert question_view(resolved['certificate']) is None
    assert not view['permits_external_action'] and not view['permits_notification']


def test_all_machine_codes_have_bilingual_display_labels():
    for language in ('ro','uk'):
        assert set(PROMPTS[language]) == set(OPTIONS[language]) == set(ONTOLOGY)
        for field,labels in ONTOLOGY.items():
            assert set(OPTIONS[language][field]) == set(labels)
            assert all(isinstance(v,str) and v.strip() for v in OPTIONS[language][field].values())


def test_tampered_certificate_cannot_supply_display_options():
    a = {field:labels[0] for field,labels in ONTOLOGY.items()}
    certificate = assure(dict(language='uk',hypotheses=[a,dict(a,concern='1')],context_digest=digest('fixture')))
    certificate['tree']['branches']['UNREGISTERED'] = {'action':'CONTINUE'}
    with pytest.raises(ValueError): question_view(certificate)
