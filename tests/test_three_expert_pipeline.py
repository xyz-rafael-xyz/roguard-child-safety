from types import SimpleNamespace
import pytest
from roguard.three_expert_pipeline import ThreeExpertPipeline


class Component:
    def __init__(self, identity):self.identity=identity;self.calls=0
    def predict_many(self, payloads, **kwargs):
        self.calls+=1
        return [{'component':self.identity} for _ in payloads]
    def predict(self, payload, **kwargs):
        self.calls+=1
        return {'component':self.identity}


def pipeline():
    fusion=SimpleNamespace(identity='fusion',_spec=dict(language='ro',native_artifact_sha256='n',
        field_artifact_sha256='s',interpreter_archive_sha256='i'),
        predict=lambda p,n,s,i,focus:dict(parts=[n,s,i],focus=focus))
    parts=[Component(k) for k in ('n','s','i')]
    return ThreeExpertPipeline(fusion,*parts),parts


def test_runs_all_components_and_preserves_focus():
    model,parts=pipeline()
    result=model.predict(dict(language='ro',turns=[dict(role='minor',text='Bună.')]),focus=0)
    assert result['focus']==0
    assert [r['component'] for r in result['parts']]==['n','s','i']
    assert [p.calls for p in parts]==[1,1,1]


@pytest.mark.parametrize('change',['language','focus','empty','oversized'])
def test_invalid_batches_rejected_before_inference(change):
    model,parts=pipeline();p=dict(language='ro',turns=[dict(role='minor',text='Bună.')])
    rows=[p,p];focuses=[0,0]
    if change=='language':rows=[p,dict(p,language='uk')]
    if change=='focus':focuses=[0,99]
    if change=='empty':rows=[];focuses=[]
    if change=='oversized':rows=[p]*129;focuses=[0]*129
    with pytest.raises(ValueError):model.predict_many(rows,focuses=focuses)
    assert not any(p.calls for p in parts)


@pytest.mark.parametrize('index',[0,1,2])
def test_all_three_model_identities_are_pinned(index):
    model,parts=pipeline();parts[index].identity='other'
    with pytest.raises(ValueError):ThreeExpertPipeline(model.fusion,*parts)


def test_incomplete_semantic_batch_cannot_drop_a_case():
    model,parts=pipeline();parts[1].predict_many=lambda *a,**k:[]
    with pytest.raises(ValueError):model.predict(dict(language='ro',turns=[dict(role='minor',text='Bună.')]))
    assert parts[2].calls==0


@pytest.mark.parametrize('asset',['../outside','/outside'])
def test_bundle_paths_validated_before_construction(tmp_path,monkeypatch,asset):
    import hashlib,json
    from roguard import three_expert_pipeline as module
    spec=dict(format='roguard-three-expert-bundle-v1',fusion=dict(path='fusion.json',sha256='a'*64),
        components=dict(native=dict(artifact='n.json',checkpoint='n.safetensors',base='base',manifest='manifest.json'),
            semantic=dict(heads='heads.json.gz',artifact='s.json',checkpoint='s.safetensors',base='base',manifest='manifest.json'),
            interpreter=dict(archive='adapter.json.gz',base=asset)))
    p=tmp_path/'bundle.json';p.write_text(json.dumps(spec))
    monkeypatch.setattr(module,'ThreeExpertReview',lambda *a,**k:pytest.fail('Constructed invalid bundle'))
    with pytest.raises(ValueError):module.ThreeExpertPipeline.from_bundle(p,expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def test_unified_loader_dispatches_three_expert_bundle(tmp_path,monkeypatch):
    import hashlib,json
    from roguard import model_bundle
    p=tmp_path/'bundle.json';p.write_text(json.dumps(dict(format='roguard-three-expert-bundle-v1')))
    monkeypatch.setattr(model_bundle.ThreeExpertPipeline,'from_bundle',lambda *a,**k:'combined')
    assert model_bundle.load_bundle(p,expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest())=='combined'
