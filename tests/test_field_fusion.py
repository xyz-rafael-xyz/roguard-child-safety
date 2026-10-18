import hashlib
import json

import pytest

from roguard.assurance import policy_digest
from roguard.evidence import ONTOLOGY
from roguard.field_fusion import FieldReviewFusion
from roguard.native_review import prepare_input


def setup(tmp_path):
    spec=dict(format='roguard-field-fusion-artifact-v1',language='ro',policy_digest=policy_digest(),
              native_heads_sha256='a'*64,pretrained_heads_sha256='b'*64,
              native_weights={f:1. for f in ONTOLOGY},couplings=None,threshold=.5)
    spec['native_weights']['assertion']=0.
    path=tmp_path/'fusion.json';path.write_text(json.dumps(spec));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    model=FieldReviewFusion(path,expected_sha256=digest)
    payload=dict(language='ro',turns=[dict(role='minor',text='Un exemplu fictiv pentru verificare.')])
    _,focus,identity=prepare_input(payload,None)
    labels=dict(subject='SELF',time='CURRENT',assertion='AFFIRMED',source='DIRECT',relation='NEW',concern='1')
    reports=[]
    for source in ('a','b'):
        if source=='b':labels['assertion']='NEGATED'
        reports.append(dict(format='roguard-native-fields-v1',language='ro',focus=focus,input_digest=identity,
            model_identity=source*64,policy_digest=policy_digest(),score=.5,predicted=1,
            field_probabilities={f:[float(v==labels[f]) for v in values] for f,values in ONTOLOGY.items()}))
    return model,payload,reports,path


def test_assertion_can_come_from_other_representation(tmp_path):
    model,payload,reports,_=setup(tmp_path)
    value=model.predict(payload,*reports)
    assert value['predicted']==0
    assert value['score']==0
    assert not value['permits_notification'] and not value['establishes_safety']
    assert value['probabilities_are_calibrated'] is False


def test_abstaining_parent_remains_abstention(tmp_path):
    model,payload,reports,_=setup(tmp_path)
    reports[0].update(field_probabilities=None,score=None,predicted=None)
    result=model.predict(payload,*reports)
    assert result['predicted'] is None and result['review_requested']


@pytest.mark.parametrize('key,value', [('model_identity','c'*64),('input_digest','c'*64),
                                      ('language','uk'),('policy_digest','c'*64),('focus',1)])
def test_mismatched_report_rejected(tmp_path,key,value):
    model,payload,reports,_=setup(tmp_path);reports[0][key]=value
    with pytest.raises(ValueError):model.predict(payload,*reports)


def test_invalid_vector_rejected(tmp_path):
    model,payload,reports,_=setup(tmp_path);reports[0]['field_probabilities']['concern']=[float('nan'),1.]
    with pytest.raises(ValueError):model.predict(payload,*reports)


def test_artifact_mismatch_rejected(tmp_path):
    _,_,_,path=setup(tmp_path)
    with pytest.raises(ValueError):FieldReviewFusion(path,expected_sha256='0'*64)
