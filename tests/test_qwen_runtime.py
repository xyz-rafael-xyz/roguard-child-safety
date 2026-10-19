import hashlib
import json
import pytest
from roguard import qwen_review as runtime
from training.qwen_review_v9 import POLICY


def model(tmp_path, monkeypatch, values=None):
    spec=dict(format='roguard-qwen-review-experiment-v1',base_model=runtime.BASE_MODEL,
        base_revision=runtime.BASE_REVISION,policy_sha256=runtime.CONTRACT,
        adapter_sha256='1'*64,thresholds={'ro':.4,'uk':.6})
    path=tmp_path/'artifact.json';raw=json.dumps(spec).encode();path.write_bytes(raw)
    class Engine:
        def __init__(self,*args):self.inputs=None
        def scores(self,prefixes):
            self.inputs=prefixes
            return values if values is not None else [(.5,True)]*len(prefixes)
    monkeypatch.setattr(runtime,'_Engine',Engine)
    return runtime.QwenReviewModel(path,tmp_path/'adapter',tmp_path/'base',expected_sha256=hashlib.sha256(raw).hexdigest())


def payload(language='uk'):
    return dict(language=language,turns=[dict(role='minor',text='Привіт.')])


def test_contract_focus_and_no_side_effect_permissions(tmp_path,monkeypatch):
    assert runtime.POLICY==POLICY
    m=model(tmp_path,monkeypatch)
    p=payload();p['turns'].append(dict(role='minor',text='FUTURE_MESSAGE'))
    result=m.predict(p,focus=0)
    assert len(m._engine.inputs[0]['turns'])==1
    assert 'FUTURE_MESSAGE' not in json.dumps(result)
    assert result['predicted']==0 and not result['permits_external_action']
    assert not result['permits_notification'] and not result['establishes_safety']
    assert m.predict(payload('ro'))['predicted']==1


def test_abstention_remains_visible(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch,[(.9,False)])
    result=m.predict(payload())
    assert result['predicted'] is None and result['review_requested']
    assert result['decision']=='clarify'


def test_entire_batch_validated_before_loading(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch)
    with pytest.raises(ValueError):m.predict_many([payload(),dict(language='en',turns=[])])
    assert m._engine is None


@pytest.mark.parametrize('values',[[],[(float('nan'),True)],[(None,True)],[(.5,1)]])
def test_incomplete_or_invalid_engine_output_rejected(tmp_path,monkeypatch,values):
    m=model(tmp_path,monkeypatch,values)
    with pytest.raises(ValueError):m.predict(payload())


def test_artifact_tamper_rejected(tmp_path,monkeypatch):
    model(tmp_path,monkeypatch)
    with pytest.raises(ValueError):
        runtime.QwenReviewModel(tmp_path/'artifact.json','adapter','base',expected_sha256='0'*64)


def test_bundle_rejects_path_escape_before_model_load(tmp_path,monkeypatch):
    from roguard.model_bundle import load_bundle
    spec=dict(format='roguard-qwen-review-bundle-v1',artifact_sha256='1'*64,
              assets=dict(artifact='../outside.json',adapter='adapter',base='base'))
    path=tmp_path/'bundle.json';raw=json.dumps(spec).encode();path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_bundle(path,expected_sha256=hashlib.sha256(raw).hexdigest(),device='mps')


def test_bundle_requires_apple_device(tmp_path):
    from roguard.model_bundle import load_bundle
    raw=json.dumps(dict(format='roguard-qwen-review-bundle-v1')).encode()
    path=tmp_path/'bundle.json';path.write_bytes(raw)
    with pytest.raises(ValueError,match='Apple Silicon'):
        load_bundle(path,expected_sha256=hashlib.sha256(raw).hexdigest())
