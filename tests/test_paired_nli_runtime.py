import hashlib
import json
import threading
import pytest
from roguard.paired_nli import PairedNLIReview,prepare,HYPOTHESIS,PREPROCESSING
from roguard.nli_v21 import MODEL_ID,REVISION,BASE_HASHES


def fake(scores):
    model=object.__new__(PairedNLIReview);model.identity='f'*64;model.thresholds={'ro':.4,'uk':.6};model._lock=threading.RLock()
    class Engine:
        def scores(self,premises):return scores
    model._engine=Engine();return model


def payload(language='ro'):return dict(language=language,turns=[dict(role='minor',text='Salut')])


def test_language_thresholds_and_abstention_never_grant_actions():
    values=fake([.5,.5,None]).predict_many([payload(),payload('uk'),payload()])
    assert [v['predicted'] for v in values]==[1,0,None]
    assert values[-1]['decision']=='clarify' and values[-1]['review_requested']
    for v in values:
        assert not v['permits_external_action'] and not v['permits_notification'] and not v['establishes_safety']
        assert 'Salut' not in json.dumps(v)


@pytest.mark.parametrize('scores',[[float('nan')],[float('inf')],[-.1],[1.1],[True],[],[.5,.6]])
def test_invalid_model_outputs_fail(scores):
    with pytest.raises(ValueError):fake(scores).predict(payload())


def test_future_turns_do_not_affect_inference_or_identity():
    first=payload();future=dict(first,turns=first['turns']+[object()])
    assert prepare(first)==prepare(future,0)
    from training.train_paired_nli_v9 import premise
    row=dict(first,focus=0)
    assert premise(row)==prepare(first)[0]


def test_bad_batch_rejected_before_engine_load():
    model=fake([.5]);model._engine=None
    with pytest.raises(ValueError):model.predict_many([payload(),payload('en')])
    assert model._engine is None


@pytest.mark.parametrize('change',[{'thresholds':{'ro':.5}},{'thresholds':{'ro':True,'uk':.5}},{'score':'renormalized-without-neutral'},{'base_hashes':{}},{'hypothesis':'other'}])
def test_artifact_contract_checked_before_assets(tmp_path,change):
    spec=dict(format='roguard-paired-nli-artifact-v1',base=MODEL_ID,revision=REVISION,thresholds={'ro':.5,'uk':.5},hypothesis=HYPOTHESIS,checkpoint_sha256='a'*64,base_hashes=BASE_HASHES,score='softmax-entailment-including-neutral',preprocessing=PREPROCESSING);spec.update(change)
    raw=json.dumps(spec).encode();path=tmp_path/'artifact.json';path.write_bytes(raw)
    with pytest.raises(ValueError):PairedNLIReview(path,tmp_path/'missing',tmp_path/'missing',expected_sha256=hashlib.sha256(raw).hexdigest())
