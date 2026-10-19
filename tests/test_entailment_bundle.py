import hashlib
import json
import pytest
from roguard.entailment_bundle import load_bundle,runtime_extra,FORMAT


def write(tmp_path,spec):
    raw=json.dumps(spec).encode();path=tmp_path/'bundle.json';path.write_bytes(raw)
    return path,hashlib.sha256(raw).hexdigest()


def spec():return dict(format=FORMAT,kind='support-revision',artifact=dict(path='review.json',sha256='a'*64),checkpoint='weights.safetensors',base='base')


@pytest.mark.parametrize('kind,module,attribute',[('review','roguard.paired_nli','PairedNLIReview'),('support-revision','roguard.policy_entailment','PolicyEntailmentReview')])
def test_loads_bilingual_model_with_pinned_artifact(tmp_path,monkeypatch,kind,module,attribute):
    import importlib
    called=[]
    monkeypatch.setattr(importlib.import_module(module),attribute,lambda *a,**k:called.append((a,k)) or 'model')
    value=spec();value['kind']=kind;path,digest=write(tmp_path,value)
    assert runtime_extra(path,digest)=='semantic'
    assert load_bundle(path,expected_sha256=digest)=='model'
    assert called[0][0]==(tmp_path/'review.json',tmp_path/'weights.safetensors',tmp_path/'base')
    assert called[0][1]==dict(expected_sha256='a'*64,device='cpu')


@pytest.mark.parametrize('change',[{'kind':'guess'},{'checkpoint':'../escape'},{'base':'/tmp/escape'},{'artifact':dict(path='x',sha256='bad')},{'extra':True}])
def test_invalid_or_escaping_bundle_rejected(tmp_path,change):
    value=spec();value.update(change);path,digest=write(tmp_path,value)
    with pytest.raises(ValueError):load_bundle(path,expected_sha256=digest)


def test_checksum_checked_before_loading(tmp_path):
    path,digest=write(tmp_path,spec())
    with pytest.raises(ValueError):load_bundle(path,expected_sha256='b'*64)
