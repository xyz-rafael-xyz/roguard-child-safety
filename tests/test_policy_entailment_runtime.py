import threading
import pytest
from roguard.policy_entailment import PolicyEntailmentReview,HYPOTHESES


def fake(values):
    model=object.__new__(PolicyEntailmentReview);model._identity='a'*64;model._thresholds={'ro':.4,'uk':.6};model._lock=threading.RLock()
    class Engine:
        def scores(self,premises):return values
    model._engine=Engine();return model


def payload(language='ro'):return dict(language=language,turns=[dict(role='minor',text='hello')])


def test_support_and_revision_are_distinct_and_grant_no_authority():
    values=[[.7,.1],[.1,.7],[.7,.7],[.1,.1],None]
    result=fake(values).predict_many([payload()]*len(values))
    assert [v['predicted'] for v in result]==[1,1,1,0,None]
    assert [v['reason'] for v in result]==['PERSONAL_SUPPORT_PROPOSAL','REVISION_REVIEW_PROPOSAL','SUPPORT_AND_REVISION_PROPOSAL','NO_SIGNAL_DETECTED','MODEL_CONTEXT_LIMIT']
    for value in result:
        assert not value['permits_external_action'] and not value['permits_notification'] and not value['establishes_safety']


@pytest.mark.parametrize('value',[[.5],[.5,.5,.5],[True,.5],[float('nan'),.5],[-.1,.5],[1.1,.5],'0.5'])
def test_malformed_scores_rejected(value):
    with pytest.raises(ValueError):fake([value]).predict(payload())


def test_same_training_hypotheses_and_read_only_settings():
    from training.policy_entailment_v9 import HYPOTHESES as TRAINING
    assert HYPOTHESES==TRAINING
    model=fake([[.5,.1]]);limits=model.thresholds;limits['ro']=1
    assert model.predict(payload())['predicted']==1
    assert model.predict(payload('uk'))['predicted']==0
    with pytest.raises(AttributeError):model.identity='b'*64
