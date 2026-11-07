"""Independent threshold sweep and corruption detection for retained DEV logits."""
from copy import deepcopy
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('kind', ['random', 'ties', 'endpoints', 'adjacent', 'separated'])
def test_sorted_count_frontier_matches_registered_selector(kind):
    np = pytest.importorskip('numpy')
    from eval.audit_attention_development_v10 import threshold_audit
    from training.separation_frontier_v9 import frontier
    rng = np.random.default_rng(1074)
    rows = [{'gold': i % 2, 'group': str(i // 16)} for i in range(80)]
    scores = rng.uniform(0, 1, len(rows))
    if kind == 'ties': scores = rng.choice([0., .25, .5, 1.], len(rows))
    if kind == 'endpoints': scores = rng.choice([0., 1.], len(rows))
    if kind == 'adjacent': scores = rng.choice([.5, np.nextafter(.5, 1.), np.nextafter(.5, 0.)], len(rows))
    if kind == 'separated': scores = np.array([.1+.1*rng.random() if r['gold'] == 0 else .8+.1*rng.random() for r in rows])
    actual = threshold_audit(rows, scores)
    expected = frontier([{**row, 'score': float(v)} for row, v in zip(rows, scores)])
    assert actual['threshold'] == expected['threshold']
    assert actual['score_margin'] == expected['score_margin']
    assert actual['minimax_rank'] == expected['rank'][:3]
    assert actual['errors'] == sum(int(int(v >= actual['threshold']) != row['gold']) for row, v in zip(rows, scores))


def fixture():
    np = pytest.importorskip('numpy')
    from training.native_fields_v9 import SIZES
    from training.large_review_attention_v10 import summarize
    rng = np.random.default_rng(2074)
    rows = [{'case_id': str(i), 'language': 'ro', 'gold': i % 2, 'group': str(i // 8),
             'fields': [int(rng.integers(size)) for size in SIZES]} for i in range(32)]
    root = Path(__file__).resolve().parents[1]
    couplings = json.loads((root/'research/v10-evidence/crossed-review-features-62/v9-binding/ro-review.json').read_text())['couplings']
    logits = rng.normal(size=(len(rows), sum(SIZES)+2))
    baseline = rng.integers(2, size=len(rows))
    summary, _ = summarize(rows, logits, couplings, baseline)
    record = dict(summary, epoch=2, completed_training_updates=10, completed_training_presentations=66)
    return rows, logits, couplings, baseline, record


def test_all_checkpoint_statistics_match_independent_recount():
    from eval.audit_attention_development_v10 import verify_checkpoint
    rows, logits, couplings, baseline, record = fixture()
    predictions, scores, error = verify_checkpoint(rows, logits, couplings, baseline, record,
                                                  epoch=2, train_rows=33, batch_size=8)
    assert len(predictions) == len(scores) == len(rows)
    assert error < 1e-12


@pytest.mark.parametrize('fault', ['updates', 'presentations', 'epoch', 'binary', 'fields', 'joint',
                                 'threshold', 'rank', 'introduced', 'group', 'loss'])
def test_saved_checkpoint_statistic_corruption_is_rejected(fault):
    from eval.audit_attention_development_v10 import verify_checkpoint
    rows, logits, couplings, baseline, record = fixture()
    record = deepcopy(record)
    if fault == 'updates': record['completed_training_updates'] += 1
    elif fault == 'presentations': record['completed_training_presentations'] += 1
    elif fault == 'epoch': record['epoch'] += 1
    elif fault == 'binary': record['binary_errors'] += 1
    elif fault == 'fields': record['field_errors'][0] += 1
    elif fault == 'joint': record['joint_argmax_errors'] += 1
    elif fault == 'threshold': record['threshold'] += .01
    elif fault == 'rank': record['selection_rank'][0] += 1
    elif fault == 'introduced': record['introduced_v9'] += 1
    elif fault == 'group': record['groups']['0']['rows'] += 1
    elif fault == 'loss': record['mean_field_cross_entropy'] += .01
    with pytest.raises(ValueError):
        verify_checkpoint(rows, logits, couplings, baseline, record, epoch=2, train_rows=33, batch_size=8)


@pytest.mark.parametrize('fault', ['nan', 'outside', 'empty', 'single_class'])
def test_invalid_frontier_input_is_rejected(fault):
    from eval.audit_attention_development_v10 import threshold_audit
    pytest.importorskip('numpy')
    rows = [{'gold': 0, 'group': 'a'}, {'gold': 1, 'group': 'a'}]
    scores = [.2, .7]
    if fault == 'nan': scores[0] = float('nan')
    elif fault == 'outside': scores[0] = 1.1
    elif fault == 'empty': rows, scores = [], []
    elif fault == 'single_class': rows[1]['gold'] = 0
    with pytest.raises(ValueError): threshold_audit(rows, scores)
