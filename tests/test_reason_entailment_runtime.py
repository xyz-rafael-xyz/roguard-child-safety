import threading
import pytest
from roguard.reason_entailment import ReasonEntailmentReview,HYPOTHESES


def fake(values):
    model=object.__new__(ReasonEntailmentReview);model._identity='a'*64;model._thresholds={'ro':{'support':.4,'revision':.6},'uk':{'support':.6,'revision':.8}};model._lock=threading.RLock()
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
    model=fake([[.5,.1]]);limits=model.thresholds;limits['ro']['support']=1
    assert model.predict(payload())['predicted']==1
    assert model.predict(payload('uk'))['predicted']==0
    with pytest.raises(AttributeError):model.identity='b'*64


def test_each_reason_uses_its_own_language_cutoff():
    model=fake([[.5,.5],[.5,.5],[.2,.7],[.2,.7]])
    values=model.predict_many([payload('ro'),payload('uk'),payload('ro'),payload('uk')])
    assert [v['predicted'] for v in values]==[1,0,1,0]
    assert [v['reason'] for v in values]==['PERSONAL_SUPPORT_PROPOSAL','NO_SIGNAL_DETECTED','REVISION_REVIEW_PROPOSAL','NO_SIGNAL_DETECTED']
    assert values[0]['support_threshold']==.4 and values[0]['revision_threshold']==.6


@pytest.mark.parametrize('limits',[
    {'ro':.5,'uk':.5},
    {'ro':{'support':.5},'uk':{'support':.5,'revision':.5}},
    {'ro':{'support':True,'revision':.5},'uk':{'support':.5,'revision':.5}},
    {'ro':{'support':.5,'revision':float('nan')},'uk':{'support':.5,'revision':.5}},
])
def test_malformed_reason_thresholds_fail_before_model_loading(tmp_path,limits):
    import hashlib,json
    from roguard.nli_v21 import MODEL_ID,REVISION,BASE_HASHES
    from roguard.paired_nli import PREPROCESSING
    spec=dict(format='roguard-reason-entailment-artifact-v1',base=MODEL_ID,revision=REVISION,
              thresholds=limits,hypotheses=list(HYPOTHESES),checkpoint_sha256='a'*64,
              base_hashes=BASE_HASHES,score='separate-entailment-thresholds-including-neutral',preprocessing=PREPROCESSING)
    raw=json.dumps(spec).encode();path=tmp_path/'review.json';path.write_bytes(raw)
    with pytest.raises(ValueError,match='threshold'):
        ReasonEntailmentReview(path,tmp_path/'missing-weights',tmp_path/'missing-base',expected_sha256=hashlib.sha256(raw).hexdigest())
