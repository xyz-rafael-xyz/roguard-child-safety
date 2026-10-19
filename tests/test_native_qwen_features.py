import pytest
from training.native_qwen_fusion_v9 import features


def test_features_do_not_use_labels_or_text():
    n=dict(score=.2,gold=0,text='first')
    q=dict(score=.8,valid_label_token=True,gold=0,text='first')
    baseline=features(n,q)
    n.update(gold=1,text='changed');q.update(gold=1,text='changed')
    assert features(n,q)==baseline and len(baseline)==7


def test_invalid_qwen_token_is_abstention():
    assert features(dict(score=.2),dict(score=.9,valid_label_token=False)) is None


@pytest.mark.parametrize('value',[True,float('nan'),-1,2])
def test_nonprobability_component_scores_rejected(value):
    with pytest.raises(ValueError):
        features(dict(score=value),dict(score=.9,valid_label_token=True))
