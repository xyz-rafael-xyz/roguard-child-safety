from copy import deepcopy
import pytest
from roguard.review_workspace import review_workspace


def problem(language='ro'):
    shared=dict(time='CURRENT',assertion='AFFIRMED',relation='NEW',concern='1')
    return dict(language=language,context_digest='0'*64,hypotheses=[
        dict(shared,subject='SELF',source='DIRECT'),dict(shared,subject='FICTIONAL',source='FICTION')])


@pytest.mark.parametrize('language',['ro','uk'])
def test_local_question_answer_and_recompute(language):
    planned=review_workspace(dict(operation='plan',problem=problem(language)))
    certificate=planned['certificate'];question=planned['question']
    assert certificate['status']=='CLARIFICATION_REQUIRED'
    assert question['language']==language
    assert not planned['permits_external_action']
    answered=review_workspace(dict(operation='answer',certificate=certificate,
        answers={question['field']:question['options'][0]['value']},binding=question['certificate_digest']))
    assert answered['certificate']['status']=='ACTION_DETERMINED'
    assert answered['question'] is None
    assert not answered['reviewer_authenticated']
    replay=review_workspace(dict(operation='verify',certificate=answered['certificate']))
    assert replay['certificate']==answered['certificate']


def test_stale_binding_tampering_and_unsupported_operations():
    value=review_workspace(dict(operation='plan',problem=problem()));certificate=value['certificate'];q=value['question']
    with pytest.raises(ValueError):review_workspace(dict(operation='answer',certificate=certificate,
        answers={q['field']:q['options'][0]['value']},binding='f'*64))
    modified=deepcopy(certificate);modified['status']='ACTION_DETERMINED'
    with pytest.raises(ValueError):review_workspace(dict(operation='verify',certificate=modified))
    with pytest.raises(ValueError):review_workspace(dict(operation='notify',problem=problem()))


def test_unavailable_answer_requires_reinterpretation():
    value=review_workspace(dict(operation='plan',problem=problem()));certificate=value['certificate']
    result=review_workspace(dict(operation='answer',certificate=certificate,answers={'subject':'OTHER'},
        binding=certificate['certificate_digest']))
    assert result['result']['status']=='OUTSIDE_HYPOTHESES'
    assert result['certificate'] is None and result['question'] is None
    assert not result['permits_notification']
