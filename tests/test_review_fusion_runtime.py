import hashlib
import json
import random

import pytest
from roguard.evidence import ONTOLOGY, disposition
from roguard.interpreter import CONTRACT_DIGEST
from roguard.native_review import prepare_input
from roguard.review_fusion import DIMENSION, FEATURE_DIGEST, ReviewFusion, feature_vector
from training.review_fusion_v9 import features, probability


@pytest.fixture
def sample(tmp_path):
    payload = dict(language='ro', turns=[dict(role='minor', text='Temă pentru școală.')])
    identity = prepare_input(payload)[2]
    flags = dict(language='ro', focus=0, input_digest=identity, text_in_report=False,
                 establishes_safety=False, permits_external_action=False, permits_notification=False)
    native = dict(flags, format='roguard-native-review-v1', model_identity='a'*64,
                  score=.4, predicted=0, score_is_calibrated=False)
    values = {key:labels[0] for key,labels in ONTOLOGY.items()}
    interpreter = dict(flags, format='roguard-interpreter-v1', model_identity='b'*64, contract_digest=CONTRACT_DIGEST,
        predicted=0, probabilities_are_calibrated=False, valid_label_tokens=True, interpretation=values,
        conditional_field_probabilities={key:[1.]+[0.]*(len(labels)-1) for key,labels in ONTOLOGY.items()})
    spec = dict(format='roguard-review-fusion-experiment-v1', language='ro', native_artifact_sha256='a'*64,
        interpreter_archive_sha256='b'*64, interpreter_contract_digest=CONTRACT_DIGEST,
        feature_contract_digest=FEATURE_DIGEST, selection_sha256='c'*64, threshold=.5,
        model=dict(mean=[0.]*DIMENSION, scale=[1.]*DIMENSION, coefficients=[-2.]+[.2]*DIMENSION))
    path = tmp_path/'fusion.json'; path.write_text(json.dumps(spec))
    model = ReviewFusion(path, expected_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return payload,native,interpreter,spec,model


def test_runtime_features_and_scores_match_frozen_research(sample):
    payload,native,interpreter,spec,model = sample
    rng = random.Random(938)
    for score in [0.,1.,1e-9,.4]+[rng.random() for _ in range(20)]:
        native['score'] = score
        for field,labels in ONTOLOGY.items():
            values = [rng.random() for _ in labels]; total = sum(values)
            p = [v/total for v in values]
            interpreter['conditional_field_probabilities'][field] = p
            interpreter['interpretation'][field] = labels[max(range(len(p)),key=p.__getitem__)]
        interpreter['predicted'] = int(disposition({k:{'label':v} for k,v in interpreter['interpretation'].items()})[0]!='CONTINUE')
        vector = features(native,interpreter)
        assert feature_vector(score,interpreter) == vector
        result = model.predict(payload,native,interpreter)
        assert result['score'] == probability(spec['model'],vector)
        assert not result['permits_external_action'] and not result['permits_notification']


@pytest.mark.parametrize('component',['native','interpreter'])
def test_either_abstention_stays_an_abstention(sample,component):
    payload,native,interpreter,_,model = sample
    if component == 'native': native.update(score=None,predicted=None)
    else: interpreter.update(valid_label_tokens=False,predicted=None)
    result = model.predict(payload,native,interpreter)
    assert result['predicted'] is None and result['review_requested']


@pytest.mark.parametrize('field',['model_identity','input_digest','contract_digest'])
def test_foreign_or_stale_interpreter_report_rejected(sample,field):
    payload,native,interpreter,_,model = sample
    interpreter[field] = 'd'*64
    with pytest.raises(ValueError): model.predict(payload,native,interpreter)


def test_future_turn_cannot_change_result(sample):
    payload,native,interpreter,_,model = sample
    before = model.predict(payload,native,interpreter)
    payload['turns'].append({'invalid':'future'})
    assert model.predict(payload,native,interpreter,focus=0) == before


def test_invalid_probability_cannot_enter_fusion(sample):
    payload,native,interpreter,_,model = sample
    interpreter['conditional_field_probabilities']['concern'] = [0.,1.]
    with pytest.raises(ValueError,match='argmax'): model.predict(payload,native,interpreter)
