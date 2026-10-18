import gzip
import hashlib
import json
from types import SimpleNamespace
import pytest
from roguard.assurance import policy_digest
from roguard.evidence import ONTOLOGY
import roguard.native_fields as runtime


def spec():
    n=sum(len(v) for v in ONTOLOGY.values())
    return dict(format='roguard-native-fields-experiment-v1',language='ro',hidden=0,dimensions=1536,
        native_artifact_sha256='a'*64,mean=[0.]*1536,scale=[1.]*1536,weights=dict(weight=[[0.]*1536 for _ in range(n)],bias=[0.]*n),
        threshold=.5,fields=list(ONTOLOGY),sizes=[len(v) for v in ONTOLOGY.values()],
        distribution='independent categorical fields; uncalibrated',policy_digest=policy_digest())


def load(tmp_path,monkeypatch,value):
    path=tmp_path/'heads.json.gz';raw=gzip.compress(json.dumps(value).encode(),mtime=0);path.write_bytes(raw)
    monkeypatch.setattr(runtime,'NativeReviewModel',lambda *a,**k:SimpleNamespace(language='ro',identity='a'*64))
    return runtime.NativeFieldsModel(path,'unused','unused','unused','unused',expected_heads_sha256=hashlib.sha256(raw).hexdigest())


def test_prefix_binding_and_context_abstention(tmp_path,monkeypatch):
    model=load(tmp_path,monkeypatch,spec())
    vectors={f:[1.]+[0.]*(len(v)-1) for f,v in ONTOLOGY.items()}
    class Engine:
        def predict(self,pairs):
            assert 'future' not in str(pairs)
            return [vectors,None]
    model._engine=Engine()
    payload=dict(language='ro',turns=[dict(role='minor',text='Exemplu fictiv.'),dict(role='assistant',text='future')])
    first,second=model.predict_many([payload,payload],focuses=[0,0])
    assert first['score']==runtime.factorized_review_mass(vectors)
    assert first['input_digest']==second['input_digest']
    assert second['predicted'] is None and second['review_requested'] is True
    assert not first['permits_external_action'] and first['field_independence_assumed']


@pytest.mark.parametrize('change',[lambda v:v.update(policy_digest='0'*64),
    lambda v:v['weights'].update(bias=[True]*17),lambda v:v['scale'].__setitem__(0,0),
    lambda v:v['weights']['weight'][0].pop(),lambda v:v.update(threshold=float('nan'))])
def test_invalid_artifact_rejected_before_model_loading(tmp_path,monkeypatch,change):
    value=spec();change(value)
    with pytest.raises(ValueError):load(tmp_path,monkeypatch,value)
