import pytest
from training.qualify_qwen_v9 import audit


def fixture():
    rows,predictions=[],[]
    for split in ('dev','cal'):
        for lang in ('ro','uk'):
            for source in ('formal','v7_training_frame_projection'):
                for label in (0,1):
                    key=f'{split}-{lang}-{source}-{label}'
                    rows.append(dict(case_id=key,language=lang,source=source,qualification_split=split,
                        attributes=dict(subject='SELF',time='CURRENT',assertion='AFFIRMED',source='DIRECT',relation='NEW',concern=str(label))))
                    predictions.append(dict(case_id=key,language=lang,gold=label,score=.8 if label else .2,valid_label_token=True))
    return rows,predictions


def test_calibration_never_moves_threshold_and_abstentions_fail():
    rows,predictions=fixture()
    initial=audit(rows,predictions)
    assert initial['observed_targets_pass']
    predictions[-1]['score']=.1
    changed=audit(rows,predictions)
    assert changed['thresholds']==initial['thresholds']
    assert not changed['observed_targets_pass']
    predictions[-1]['score']=.8
    predictions[-1]['valid_label_token']=False
    assert not audit(rows,predictions)['observed_targets_pass']


def test_identity_and_nonfinite_results_rejected():
    rows,predictions=fixture()
    with pytest.raises(ValueError):audit(rows,predictions[::-1])
    predictions[-1]['score']=float('nan')
    with pytest.raises(ValueError):audit(rows,predictions)
