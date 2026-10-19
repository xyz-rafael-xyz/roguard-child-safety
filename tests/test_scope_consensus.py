import hashlib
import json
from pathlib import Path
import pytest
from test_three_expert_runtime import sample
from roguard.assurance import policy_digest
from roguard.scope_consensus import CONTRACT_DIGEST, ScopeConsensusReview


def wrap(tmp_path,base,**changes):
    spec=dict(format='roguard-scope-consensus-artifact-v1',base_artifact_sha256=base.identity,
        native_threshold=.5,base_threshold=base._spec['threshold'],rescue_threshold=.85,
        selection_sha256='d'*64,policy_digest=policy_digest(),contract_digest=CONTRACT_DIGEST)
    spec.update(changes);path=tmp_path/'consensus.json';path.write_text(json.dumps(spec))
    return ScopeConsensusReview(path,base,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def eligible(sample):
    payload,native,semantic,interpreter,base,_=sample
    base._spec['model']['trees'][0]['value']=[-20.]
    native.update(score=.9,predicted=1)
    # The semantic concern head can miss the signal while its scope is useful.
    interpreter['interpretation']=dict(interpreter['interpretation'],concern='1')
    interpreter['conditional_field_probabilities']=dict(interpreter['conditional_field_probabilities'],concern=[0.,1.])
    interpreter['predicted']=1
    return payload,native,semantic,interpreter,base


def test_confident_scope_agreement_can_recover_suppressed_concern(sample,tmp_path):
    payload,n,s,i,base=eligible(sample);model=wrap(tmp_path,base)
    i['conditional_field_probabilities']['subject']=[.75,0.,.25]
    result=model.predict(payload,n,s,i)
    assert result['predicted']==1 and result['rescue_applied']
    assert result['base_score']<.5 and result['score']>.5
    assert not result['score_is_calibrated'] and not result['permits_notification']


def test_existing_positive_is_preserved_without_consensus(sample,tmp_path):
    payload,n,s,i,base,_=sample;model=wrap(tmp_path,base)
    value=model.predict(payload,n,s,i)
    assert value['predicted']==1 and not value['rescue_applied']


def test_component_abstention_cannot_be_rescued(sample,tmp_path):
    payload,n,s,i,base=eligible(sample);model=wrap(tmp_path,base)
    s.update(field_probabilities=None,interpretation=None,score=None,predicted=None)
    result=model.predict(payload,n,s,i)
    assert result['predicted'] is None and result['review_requested']


@pytest.mark.parametrize('component',[1,2,3])
def test_scope_wrapper_preserves_all_component_input_bindings(sample,tmp_path,component):
    parts=list(eligible(sample));model=wrap(tmp_path,parts[-1]);parts[component]['input_digest']='e'*64
    with pytest.raises(ValueError):model.predict(*parts[:4])


def test_semantic_fiction_disagreement_blocks_rescue(sample,tmp_path):
    payload,n,s,i,base=eligible(sample);model=wrap(tmp_path,base)
    s['interpretation']=dict(s['interpretation'],subject='FICTIONAL',source='FICTION')
    s['field_probabilities']=dict(s['field_probabilities'],subject=[0.,0.,1.],source=[0.,0.,1.])
    result=model.predict(payload,n,s,i)
    assert result['predicted']==0 and not result['rescue_applied']


@pytest.mark.parametrize('key,value',[('native_threshold',float('nan')),('rescue_threshold',True),
                                     ('base_threshold',.7),('base_artifact_sha256','f'*64)])
def test_invalid_configuration_rejected(sample,tmp_path,key,value):
    with pytest.raises(ValueError):wrap(tmp_path,sample[4],**{key:value})


def test_runtime_matches_every_actual_stress_component_record(tmp_path):
    from roguard.three_expert import ThreeExpertReview
    root=Path(__file__).resolve().parents[1]/'research/v9-stabilization'
    source=root/'stress-components-106/actual'
    if not (source/'report.json').exists():pytest.skip('Actual component evidence unavailable')
    read=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
    selection=json.loads((root/'consensus-rescue-113/selection.json').read_text())
    path=root/'installed-repaired-99/fusion.json'
    base=ThreeExpertReview(path,expected_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    model=wrap(tmp_path,base,native_threshold=selection['native_threshold'],rescue_threshold=selection['rescue_threshold'])
    rows=read(root/'installed-repaired-99/stress.jsonl');reference=read(root/'consensus-rescue-113/stress-predictions.jsonl')
    parts=[read(source/(name+'.jsonl')) for name in ('native','semantic','interpreter')]
    for row,expected,n,s,i in zip(rows,reference,*parts):
        result=model.predict(dict(language=row['language'],turns=row['turns']),n,s,i,focus=row['focus'])
        assert result['predicted']==expected['predicted']
        assert abs(result['score']-expected['score'])<1e-12
