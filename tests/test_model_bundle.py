import hashlib
import json
import pytest

from roguard import model_bundle as bundle


def manifest(tmp_path, value):
    path=tmp_path/'bundle.json';path.write_text(json.dumps(value))
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_binary_bundle_loads_only_confined_assets(tmp_path,monkeypatch):
    value=dict(format='roguard-native-review-bundle-v1',artifact_sha256='a'*64,
        assets=dict(artifact='review.json.gz',checkpoint='model.safetensors',base='base',manifest='base.json'))
    path,digest=manifest(tmp_path,value);calls=[]
    monkeypatch.setattr(bundle,'NativeReviewModel',lambda *args,**kwargs:calls.append((args,kwargs)))
    bundle.load_bundle(path,expected_sha256=digest)
    assert calls==[(tuple(tmp_path/p for p in ('review.json.gz','model.safetensors','base','base.json')),
                   dict(expected_artifact_sha256='a'*64,device='cpu'))]


@pytest.mark.parametrize('format',['roguard-native-field-bundle-v1','roguard-field-bundle-v1'])
def test_tampering_is_rejected_before_dispatch(tmp_path,monkeypatch,format):
    path,digest=manifest(tmp_path,dict(format=format));path.write_bytes(path.read_bytes()+b' ')
    monkeypatch.setattr(bundle,'load_fields',lambda *a,**kw:pytest.fail('Dispatched tampered bundle'))
    monkeypatch.setattr(bundle.FieldReviewPipeline,'from_bundle',lambda *a,**kw:pytest.fail('Dispatched tampered bundle'))
    with pytest.raises(ValueError,match='identity'):bundle.load_bundle(path,expected_sha256=digest)


def test_binary_bundle_cannot_escape_via_symlink(tmp_path):
    (tmp_path/'base').symlink_to(tmp_path.parent,target_is_directory=True)
    path,digest=manifest(tmp_path,dict(format='roguard-native-review-bundle-v1',artifact_sha256='a'*64,
        assets=dict(artifact='review.json.gz',checkpoint='model.safetensors',base='base',manifest='base.json')))
    with pytest.raises(ValueError,match='escapes'):bundle.load_bundle(path,expected_sha256=digest)


@pytest.mark.parametrize('value',[{},[],dict(format='unknown'),dict(format='roguard-native-review-bundle-v1',assets={})])
def test_unrecognized_contract_fails_closed(tmp_path,value):
    path,digest=manifest(tmp_path,value)
    with pytest.raises(ValueError):bundle.load_bundle(path,expected_sha256=digest)
