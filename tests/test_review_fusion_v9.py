import pytest
from roguard.evidence import ONTOLOGY
from training.review_fusion_v9 import features, folds, probability, threshold, metrics


def structured():
    return dict(valid_label_tokens=True, interpretation={k: v[0] for k,v in ONTOLOGY.items()},
                conditional_field_probabilities={k: [1.0]+[0.0]*(len(v)-1) for k,v in ONTOLOGY.items()})


def test_features_never_use_gold_interpretation_or_case_metadata():
    record = structured()
    before = features(dict(score=.2), record)
    record.update(gold=1, attributes={'concern': '1'}, case_id='arbitrary', source='arbitrary')
    assert features(dict(score=.2, gold=0), record) == before
    record['valid_label_tokens'] = False
    assert features(dict(score=.2), record) is None


def test_malformed_probabilities_are_rejected():
    record = structured(); record['conditional_field_probabilities']['concern'] = [float('nan'), 0]
    with pytest.raises(ValueError, match='probability vector'):
        features(dict(score=.2), record)


def test_connected_pairs_and_same_focal_stay_in_one_fold():
    def row(identity, text, pair):
        return dict(case_id=identity, language='ro', focus=0, pair_group=pair,
                    turns=[dict(role='minor', text=text)])
    assignments = folds([row('a','one','pair1'), row('b','two','pair1'), row('c','two','pair2')])
    assert len(set(assignments)) == 1


def test_invalid_structured_outputs_cannot_be_hidden_by_threshold():
    rows = [dict(source=s) for s in ('formal','formal','v7_training_frame_projection','v7_training_frame_projection')]
    truth = [1,0,1,0]; scores = [None,.1,.9,.1]
    limit = threshold(rows, truth, scores)
    assert not metrics(rows, truth, scores, limit)['formal']['passes']
    assert probability({}, None) is None
