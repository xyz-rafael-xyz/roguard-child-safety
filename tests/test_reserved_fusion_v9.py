import json
import pytest

from training.reserved_fusion_v9 import require_counts, qualify, sha


def groups():
    return {name: dict(positive=100,negative=100,missed=1,false_reviews=5,abstentions=0,passes=True)
            for name in ('formal','conversational')}


def test_exact_accuracy_boundary_and_abstentions():
    value=groups();require_counts(value)
    value['formal']['abstentions']=6;require_counts(value)
    value['formal']['abstentions']=7
    with pytest.raises(ValueError):require_counts(value)


@pytest.mark.parametrize('field,value', [('missed',2),('false_reviews',6),('positive',0),('negative',0),
    ('missed',-1),('missed',False),('passes',False),('abstentions',None)])
def test_boolean_pass_cannot_override_failed_counts(field,value):
    records=groups();records['conversational'][field]=value
    with pytest.raises(ValueError):require_counts(records)


def test_both_strata_required():
    value=groups();del value['formal']
    with pytest.raises(ValueError):require_counts(value)


@pytest.fixture
def evidence(tmp_path):
    def write(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return sha(path)
    archive=write('interpreter-14/selected-adapter.json.gz',{})
    write('interpreter-full-21/selection.json',dict(selected_study=14,archive_sha256=archive))
    ir=write('installed-interpreter-30/screen/report.json',dict(parity_pass=True,archive_sha256=archive,
        decision_changes=0,schema_changes=0,interpretation_changes=0,unexpected_context_abstentions=0,
        maximum_probability_error=.0001,runtime_source_sha256='a'*64,wheel_sha256='b'*64))
    model=dict(mean=[0],scale=[1],coefficients=[0,1])
    for language in ('ro','uk'):
        select=write('review-fusion-36/'+language+'/selection.json',dict(model=model,selected=dict(metrics=groups(),threshold=.5)))
        write('review-fusion-36/'+language+'/audit.json',dict(calibration=groups(),counts_pass=True,selection_sha256=select))
        native=write('native-full-22-'+language+'/review.json.gz',{})
        artifact=write('review-fusion-38/'+language+'/fusion.json',dict(model=model,threshold=.5,
            selection_sha256=select,native_artifact_sha256=native,interpreter_archive_sha256=archive))
        nr=write('installed-native-29/'+language+'/report.json',dict(complete=True,parity_pass=True,artifact_sha256=native,
            runtime_source_sha256='c'*64,splits={s:dict(parity_pass=True,abstentions=0,decision_disagreements=0,max_score_error=.00001)
                                                for s in ('dev','cal')}))
        write('review-fusion-38/'+language+'/runtime-report.json',dict(parity_pass=True,decision_changes=0,
            maximum_score_error=.0001,actual_installed_neural_outputs_used=True,artifact_sha256=artifact,
            native_runtime_report_sha256=nr,interpreter_runtime_report_sha256=ir,wheel_sha256='b'*64,
            installed_runtime_sha256='d'*64))
    return tmp_path


def test_freeze_requires_both_languages_and_preserves_identities(evidence):
    plan=qualify(evidence)
    assert set(plan['candidates'])=={'ro','uk'}
    assert plan['stable_release_ready'] is False
    assert len(plan['verifier_source_sha256'])==4
    (evidence/'review-fusion-36/uk/audit.json').unlink()
    with pytest.raises(FileNotFoundError):qualify(evidence)


@pytest.mark.parametrize('relative,key,value',[
    ('installed-interpreter-30/screen/report.json','decision_changes',1),
    ('review-fusion-36/uk/audit.json','counts_pass',False),
    ('review-fusion-38/ro/fusion.json','threshold',.2),
    ('review-fusion-38/uk/runtime-report.json','actual_installed_neural_outputs_used',False),
    ('installed-native-29/uk/report.json','parity_pass',False),
])
def test_changed_evidence_cannot_qualify(evidence,relative,key,value):
    path=evidence/relative;record=json.loads(path.read_text());record[key]=value;path.write_text(json.dumps(record))
    with pytest.raises(ValueError):qualify(evidence)
