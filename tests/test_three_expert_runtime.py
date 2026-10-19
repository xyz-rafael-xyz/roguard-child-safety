import copy
import hashlib
import json
import pytest
from roguard.assurance import policy_digest
from roguard.evidence import ONTOLOGY
from roguard.interpreter import CONTRACT_DIGEST
from roguard.native_review import prepare_input
from roguard.three_expert import DIMENSION, FEATURE_DIGEST, ThreeExpertReview, validate_trees


@pytest.fixture
def sample(tmp_path):
    payload=dict(language='ro',turns=[dict(role='minor',text='Temă pentru școală.')])
    flags=dict(language='ro',focus=0,input_digest=prepare_input(payload)[2],text_in_report=False,
        establishes_safety=False,permits_external_action=False,permits_notification=False)
    values={f:labels[0] for f,labels in ONTOLOGY.items()}
    probs={f:[1.]+[0.]*(len(labels)-1) for f,labels in ONTOLOGY.items()}
    native=dict(flags,format='roguard-native-review-v1',model_identity='a'*64,score=.4,predicted=0,score_is_calibrated=False)
    semantic=dict(flags,format='roguard-native-fields-v1',model_identity='b'*64,policy_digest=policy_digest(),
        interpretation_is_head_argmax=True,interpretation=values,field_probabilities=probs,score=.4,predicted=0,probabilities_are_calibrated=False)
    interpreter=dict(flags,format='roguard-interpreter-v1',model_identity='c'*64,contract_digest=CONTRACT_DIGEST,
        valid_label_tokens=True,interpretation=values,conditional_field_probabilities=probs,predicted=0,probabilities_are_calibrated=False)
    tree=dict(feature=[-2],threshold=[-2.],left=[-1],right=[-1],value=[.8])
    model=dict(format='roguard-boosted-score-v1',dimension=DIMENSION,intercept=0.,learning_rate=.05,trees=[tree],sklearn_version='test')
    spec=dict(format='roguard-three-expert-artifact-v1',language='ro',native_artifact_sha256='a'*64,
        field_artifact_sha256='b'*64,interpreter_archive_sha256='c'*64,interpreter_contract_digest=CONTRACT_DIGEST,
        feature_contract_digest=FEATURE_DIGEST,policy_digest=policy_digest(),selection_sha256='d'*64,model=model,threshold=.5)
    path=tmp_path/'model.json';path.write_text(json.dumps(spec))
    runtime=ThreeExpertReview(path,expected_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return payload,native,semantic,interpreter,runtime,model


def test_three_bound_reports_produce_no_external_authority(sample):
    payload,native,semantic,interpreter,model,_=sample
    result=model.predict(payload,native,semantic,interpreter)
    assert result['predicted']==1 and not result['reports_authenticated']
    assert not result['permits_notification'] and not result['permits_external_action']


@pytest.mark.parametrize('index',[1,2,3])
@pytest.mark.parametrize('key,value',[('input_digest','f'*64),('model_identity','f'*64),('focus',True),('permits_notification',True)])
def test_each_component_binding_is_checked(sample,index,key,value):
    parts=list(sample);parts[index][key]=value
    with pytest.raises(ValueError):parts[4].predict(*parts[:4])


def test_semantic_abstention_cannot_be_overridden(sample):
    payload,native,semantic,interpreter,model,_=sample
    semantic.update(field_probabilities=None,interpretation=None,score=None,predicted=None)
    result=model.predict(payload,native,semantic,interpreter)
    assert result['predicted'] is None and result['review_requested']


@pytest.mark.parametrize('change',['cycle','nan','dimension','unreachable'])
def test_unsafe_tree_structures_rejected(sample,change):
    model=copy.deepcopy(sample[-1]);tree=model['trees'][0]
    if change=='cycle':tree.update(feature=[0],left=[0],right=[0])
    elif change=='nan':tree['value']=[float('nan')]
    elif change=='dimension':model['dimension']=1
    else:
        for key in tree:tree[key]*=2
    with pytest.raises(ValueError):validate_trees(model)
