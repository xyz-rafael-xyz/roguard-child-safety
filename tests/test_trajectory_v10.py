from copy import deepcopy
import gzip
import json

import pytest

from roguard.assurance import digest
from roguard.evidence import ONTOLOGY
from roguard.flow import FlowLedger
from roguard.trajectory_data import FORMAT, audit_splits, prefix, input_identity, validate_case
from roguard.trajectory_evaluation import evaluate, paired_compare
from roguard.trajectory_memory import select, verify_view, derive_memory, units
from roguard.trajectory_model import DIMENSION, RESPONSE_LABELS, TrajectoryModel, TrajectorySession, features
from roguard.trajectory_review import annotation_packet, validate_annotations, reviewer_packet, score_reviewer
from roguard.trajectory_sets import WORLDS, calibrate, predict_set
from roguard.trajectory_gate import assess
from roguard.trajectory_neural import factors_from_logits, view_payload
from roguard.trajectory_study import freeze,open_once
from roguard.trajectory_data import write_jsonl


def case(name='a', review=1, split='challenge'):
    return {'format':FORMAT,'case_id':name,'language':'ro','turns':[{'role':'minor','text':'Am nevoie de ajutor.'}],
            'split':split,'family':name,'author':'author-'+name,'generator':'generator-'+name,'synthetic':True,
            'checkpoints':[{'focus':0,'review':review,'world':WORLDS[0],
                            'evidence':[{'turn':0,'start':0,'end':19}]}]}


def prediction(c,value):
    return {'checkpoint_id':c['case_id']+':0','language':'ro','focus':0,
            'input_digest':input_identity(prefix(c,0)),'predicted':value,'latency_seconds':0.01}


def test_prefix_hides_future_labels_and_rejects_bad_offsets():
    c=case();c['turns'].append({'role':'assistant','text':'Future instruction'})
    assert prefix(c,0)=={'language':'ro','turns':c['turns'][:1]}
    c['checkpoints'][0]['evidence'][0]['turn']=1
    with pytest.raises(ValueError,match='future'):validate_case(c)


def test_no_family_or_normalized_prefix_leakage():
    a=case('a',split='train');b=case('b',split='test')
    with pytest.raises(ValueError,match='overlaps'):audit_splits([a,b])
    b['turns'][0]['text']='Un mesaj diferit pentru studiu.';b['family']='a'
    with pytest.raises(ValueError,match='family'):audit_splits([a,b])
    b['family']='b';b['author']=a['author']
    with pytest.raises(ValueError,match='author'):audit_splits([a,b],independent_test=True)


def test_memory_offsets_budget_pinning_and_source_tampering():
    p={'language':'ro','turns':[{'role':'minor','text':'  Știu. A doua propoziție!'},
                               {'role':'assistant','text':'x'*180},{'role':'minor','text':'Acum?'}]}
    v=select(p,lambda text,*args: 1 if 'Știu' in text else 0,budget=128)
    assert verify_view(p,v)
    assert any(s['text']=='Știu.' for s in v['spans'])
    assert v['consulted_turns']==[0,1,2]
    assert v['used_characters']<=128
    overflow=select(p,strategy='recent',budget=128,pinned=[1])
    assert overflow['mandatory_overflow'] and not overflow['spans']
    changed=deepcopy(v);changed['spans'][0]['text']='Invented'
    with pytest.raises(ValueError):verify_view(p,changed)
    with pytest.raises(ValueError):select(p,lambda *_:float('nan'))


def test_derived_memory_inherits_omitted_source_revocation():
    p={'language':'ro','turns':[{'role':'minor','text':'x'*200},{'role':'minor','text':'Acum?'}]}
    view=select(p,strategy='recent',budget=128)
    assert {s['turn'] for s in view['spans']}=={1}
    ledger=FlowLedger()
    for i,t in enumerate(p['turns']):
        ledger.add_source('source'+str(i),t['text'])
        ledger.permission('source'+str(i),'study','reviewer','use','grant')
    derive_memory(ledger,'memory',p,view,['source0','source1'])
    assert set(ledger.describe('memory')['sources'])=={'source0','source1'}
    ledger.permission('source0','study','reviewer','use','revoke')
    assert ledger.check(['memory'],'study','reviewer','use')['decision']=='BLOCK'


def test_memory_retains_unpunctuated_lines_and_punctuation_only_focal_turns():
    p={'language':'ro','turns':[{'role':'minor','text':'Prima linie\nA doua linie\n!!!'}]}
    view=select(p,strategy='recent',budget=256)
    assert [s['text'] for s in view['spans']]==['Prima linie','A doua linie','!!!']
    assert verify_view(p,view)
    changed=deepcopy(view);removed=changed['spans'].pop(0)
    changed['used_characters']-=len(removed['text'])+32
    with pytest.raises(ValueError,match='mandatory'):verify_view(p,changed)
    changed=deepcopy(view);changed.update(spans=[],used_characters=0,mandatory_overflow=True)
    with pytest.raises(ValueError,match='overflow'):verify_view(p,changed)


def calibration_rows(n=10):
    probabilities=[0.0]*len(WORLDS);probabilities[0]=.8;probabilities[1]=.2
    return [{'case_id':str(i),'split':'calibration','language':'ro','family':str(i),
             'world':WORLDS[0],'probabilities':probabilities,'pipeline_digest':'a'*64} for i in range(n)]


def test_conformal_family_max_rank_and_small_sample_fallback():
    rows=calibration_rows(10)
    calibration=calibrate(rows,pipeline_digest='a'*64,alpha=.1)
    assert calibration['families_per_language']=={'ro':10,'uk':0}
    assert len(predict_set(rows[0]['probabilities'],calibration,language='ro',pipeline_digest='a'*64)['hypotheses'])==1
    assert len(predict_set(rows[0]['probabilities'],calibration,language='uk',pipeline_digest='a'*64)['hypotheses'])==432
    extra=deepcopy(rows[0]);extra['case_id']='correlated';extra['probabilities']=[1/432]*432
    calibration=calibrate(rows+[extra],pipeline_digest='a'*64,alpha=.1)
    assert calibration['families_per_language']['ro']==10
    assert calibration['quantiles']['ro']==pytest.approx(1-1/432)
    small=calibrate(rows[:2],pipeline_digest='a'*64,alpha=.1)
    assert small['quantiles']['ro']==1


def test_sets_bind_pipeline_and_unsupported_expands():
    rows=calibration_rows();calibration=calibrate(rows,pipeline_digest='a'*64)
    with pytest.raises(ValueError,match='pipeline'):predict_set(rows[0]['probabilities'],calibration,language='ro',pipeline_digest='b'*64)
    assert len(predict_set(rows[0]['probabilities'],calibration,language='ro',pipeline_digest='a'*64,supported=False)['hypotheses'])==432
    calibration['quantiles']['ro']=0
    with pytest.raises(ValueError,match='integrity'):predict_set(rows[0]['probabilities'],calibration,language='ro',pipeline_digest='a'*64)


def test_evaluator_keeps_misses_abstentions_and_rejects_missing_or_future():
    cases=[case('positive'),case('negative',0)]
    predictions=[prediction(cases[0],None),prediction(cases[1],None)]
    result=evaluate(cases,predictions,bootstrap=20)['groups']['ro']
    assert result['coverage']==0 and result['recall_including_abstentions']==0
    assert result['review_or_abstention_per_1000_benign_conversations']==1000
    assert result['missed_conversations']==1 and result['mean_detection_delay_turns_among_detected'] is None
    with pytest.raises(ValueError,match='Missing'):evaluate(cases,predictions[:1])
    predictions[0]['evidence']=[{'turn':1,'start':0,'end':1}]
    with pytest.raises(ValueError,match='future'):evaluate(cases,predictions)


def test_paired_comparison_and_delayed_detection():
    c=case();c['turns'] += [{'role':'minor','text':'Încă am nevoie de ajutor.'}]
    c['checkpoints'].append({'focus':1,'review':1})
    a=prediction(c,0)
    b={'checkpoint_id':'a:1','language':'ro','focus':1,'input_digest':input_identity(prefix(c,1)),'predicted':1}
    result=evaluate([c],[a,b],bootstrap=10)['groups']['ro']
    assert result['mean_detection_delay_turns_among_detected']==1
    winner=deepcopy(a);winner['predicted']=1
    comparison=paired_compare([c],[a,b],[winner,b],bootstrap=10)
    assert comparison['strict_accuracy_delta']==.5


def model_artifact(tmp_path):
    labels={**ONTOLOGY,'review':('0','1'),'relevance':('0','1'),'response':RESPONSE_LABELS}
    heads={}
    for name,values in labels.items():
        heads[name]={'classes':list(values),'weights':[[0.0]*DIMENSION for _ in values],'bias':[0.0]*len(values)}
    spec={'format':'roguard-student-v10-research1','dimension':DIMENSION,'heads':heads,'review_threshold':.5}
    path=tmp_path/'student.gz';path.write_bytes(gzip.compress(json.dumps(spec).encode()))
    return path


def test_portable_student_digest_overflow_and_session_floor(tmp_path):
    path=model_artifact(tmp_path)
    model=TrajectoryModel(path)
    with pytest.raises(ValueError,match='digest'):TrajectoryModel(path,expected_sha256='0'*64)
    p=prefix(case(),0)
    session=TrajectorySession(model);first=session.predict(p)
    assert first['predicted']==1 and first['session_review_required']
    model.threshold=.9
    p['turns'].append({'role':'minor','text':'Acum fac matematică.'})
    second=session.predict(p)
    assert second['predicted']==0 and second['session_review_required']
    with pytest.raises(ValueError):session.resolve(first['input_digest'],expected_context_digest=first['input_digest'])
    session.resolve(first['input_digest'],expected_context_digest=second['input_digest'])
    assert not session.predict(p)['session_review_required']
    p['turns'][0]['text']='Changed history'
    with pytest.raises(ValueError):session.predict(p)
    long={'language':'ro','turns':[{'role':'minor','text':'x'*200}]}
    assert model.predict(long,budget=128)['predicted'] is None
    abstained = TrajectorySession(model)
    unsupported = abstained.predict(long,budget=128)
    long['turns'].append({'role':'minor','text':'Acum fac matematică.'})
    later = abstained.predict(long,budget=512)
    assert later['predicted']==0 and later['session_review_required']
    abstained.resolve(unsupported['input_digest'], expected_context_digest=later['input_digest'])
    assert not abstained.predict(long,budget=512)['session_review_required']


def test_annotation_packets_hide_gold_and_reject_changed_inputs():
    cases=[case()];packet,key=annotation_packet(cases,annotator_id='independent-a')
    item=packet['items'][0]
    assert item['review'] is None and item['world'] is None
    completed=deepcopy(packet)
    completed['items'][0].update(review=1,world=WORLDS[0],outside_ontology=False,uncertain=False)
    labels=validate_annotations(packet,key,completed)
    assert labels['a:0']['review']==1
    completed['items'][0]['conversation']['turns'][0]['text']='Changed'
    with pytest.raises(ValueError,match='changed'):validate_annotations(packet,key,completed)


def test_reviewer_counterbalancing_and_incomplete_work():
    cases=[case('a'),case('b',0)]
    a=[prediction(c,0) for c in cases];b=[prediction(c,1) for c in cases]
    p,k=reviewer_packet(cases,a,b,reviewer_id='r1',arm=0)
    q,l=reviewer_packet(cases,a,b,reviewer_id='r2',arm=1)
    left={v['checkpoint_id']:v['condition'] for v in k['key'].values()}
    right={v['checkpoint_id']:v['condition'] for v in l['key'].values()}
    assert all(left[key]!=right[key] for key in left)
    returned=deepcopy(p)
    for row in returned['items']:row['task_complete']=False
    report=score_reviewer(p,k,returned)
    assert all(v['accuracy_including_incomplete']==0 for v in report['conditions'].values())


def test_hash_features_are_deterministic_and_normalized():
    a=features('Știu ceva.','ro')
    assert a==features('Știu ceva.','ro')
    assert sum(x*x for x in a.values())==pytest.approx(1)


def test_neural_fields_are_a_full_distribution_and_reject_nonfinite():
    from roguard.native_fields import SIZES
    p=factors_from_logits([0.0]*(sum(SIZES)+2))
    assert len(p)==432 and sum(p)==pytest.approx(1)
    assert max(p)==pytest.approx(1/432)
    with pytest.raises(ValueError):factors_from_logits([float('nan')]*(sum(SIZES)+2))


def test_complete_selected_turns_preserve_original_formatting():
    payload={'language':'ro','turns':[{'role':'minor','text':'«Prima.\nA doua!»'}]}
    view=select(payload,strategy='recent',budget=256)
    assert view_payload(payload,view,preserve_complete=True)==payload
    partial=deepcopy(view);partial['spans']=partial['spans'][:1]
    assert view_payload(payload,partial,preserve_complete=True)['turns'][0]['text']=='«Prima.'


def test_perfect_aggregate_cannot_bypass_release_requirements():
    c=case();r=prediction(c,1)
    result=assess([c],[r])
    assert result['decision']=='REJECT' and not result['promotion_pass']
    assert 'BILINGUAL_EVALUATION_MISSING' in result['engineering_failures']
    assert 'INDEPENDENT_SEALED_EVALUATION_REQUIRED' in result['external_requirements']


def test_long_memory_uses_provenance_groups_and_keeps_revocation():
    p={'language':'ro','turns':[{'role':'minor','text':'Mesaj '+str(i)+'.'} for i in range(100)]}
    view=select(p,strategy='recent',budget=128);ledger=FlowLedger()
    ids=[]
    for i,t in enumerate(p['turns']):
        name='s'+str(i);ids.append(name);ledger.add_source(name,t['text'])
        ledger.permission(name,'study','reviewer','use','grant')
    derive_memory(ledger,'long_memory',p,view,ids)
    assert len(ledger.describe('long_memory')['sources'])==100
    assert ledger.check(['long_memory'],'study','reviewer','use')['decision']=='ALLOW'
    ledger.permission('s0','study','reviewer','use','revoke')
    assert ledger.check(['long_memory'],'study','reviewer','use')['decision']=='BLOCK'


def test_final_test_is_hash_bound_and_consumed_before_failed_evaluation(tmp_path):
    model=tmp_path/'weights';model.write_text('frozen')
    test=tmp_path/'sealed.jsonl';write_jsonl(test,[case('sealed',split='test')])
    frozen=tmp_path/'freeze.json';log=tmp_path/'exposure.json'
    freeze(artifacts=[model],test_file=test,output=frozen,exposure_log=log)
    model.write_text('changed')
    with pytest.raises(ValueError,match='changed'):open_once(frozen,exposure_log=log,development_cases=[])
    assert not log.exists()
    model.write_text('frozen')
    rows,exposure=open_once(frozen,exposure_log=log,development_cases=[])
    assert len(rows)==1 and exposure['consumed_even_if_run_fails']
    with pytest.raises(ValueError,match='already exposed'):open_once(frozen,exposure_log=log,development_cases=[])
    with pytest.raises(ValueError,match='frozen path'):
        open_once(frozen,exposure_log=tmp_path/'different.json',development_cases=[])


def test_final_test_invalid_split_still_consumes_exposure(tmp_path):
    model=tmp_path/'weights';model.write_text('frozen')
    test=tmp_path/'test.jsonl';write_jsonl(test,[case(split='train')])
    frozen=tmp_path/'freeze.json';log=tmp_path/'exposure.json'
    freeze(artifacts=[model],test_file=test,output=frozen,exposure_log=log)
    with pytest.raises(ValueError,match='declared test'):open_once(frozen,exposure_log=log,development_cases=[])
    assert log.exists()
