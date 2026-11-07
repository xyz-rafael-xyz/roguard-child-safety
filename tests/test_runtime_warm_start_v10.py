"""Head-only preparation preserves frozen weights, data order and fixed decisions."""
import pytest

from training.prepare_runtime_warm_start_v10 import check_weights, compare_logits, summarize_fixed, project_rows


@pytest.fixture
def weights():
    np = pytest.importorskip('numpy')
    original = {f'heads.{lang}.{layer}.{part}':np.ones((2,), np.float32)
                for lang in ('ro', 'uk') for layer in ('first', 'last') for part in ('weight', 'bias')}
    original['base.attention'] = np.ones((3,), np.float32)
    return original, {k:v.copy() for k,v in original.items()}


def test_only_registered_language_head_may_differ(weights):
    original, candidate = weights
    candidate['heads.ro.first.weight'][0] = 2
    check_weights(original, candidate, 'ro')
    with pytest.raises(ValueError, match='frozen v9'):
        check_weights(original, candidate, 'uk')


@pytest.mark.parametrize('fault', ['body', 'other_head', 'shape', 'dtype', 'nan', 'missing', 'extra'])
def test_invalid_warm_start_is_rejected(weights, fault):
    np = pytest.importorskip('numpy')
    original, candidate = weights
    if fault == 'body': candidate['base.attention'][0] = 2
    elif fault == 'other_head': candidate['heads.uk.last.bias'][0] = 2
    elif fault == 'shape': candidate['heads.ro.first.bias'] = np.ones(3, np.float32)
    elif fault == 'dtype': candidate['heads.ro.first.bias'] = np.ones(2, np.float64)
    elif fault == 'nan': candidate['heads.ro.first.bias'][0] = np.nan
    elif fault == 'missing': candidate.pop('heads.ro.first.bias')
    else: candidate['extra'] = np.ones(2, np.float32)
    with pytest.raises(ValueError):
        check_weights(original, candidate, 'ro')


@pytest.fixture
def logits():
    np = pytest.importorskip('numpy')
    z = np.zeros((2, 19), np.float32); z[0, 17] = 2; z[1, 18] = 3
    rows = [dict(partition='dev', gold=i, fields=[0]*6) for i in (0, 1)]
    return rows, z, {'mode':'binary', 'threshold':.75}


def test_fixed_threshold_and_six_field_metrics(logits):
    rows, z, spec = logits
    summary, decisions = summarize_fixed(rows, z, spec)
    assert decisions == [0, 1] and summary['binary_errors'] == 0 and summary['joint_argmax_errors'] == 0
    z[0, 1] = 1
    summary, decisions = summarize_fixed(rows, z, spec)
    assert decisions == [0, 1] and summary['field_errors'] == [1, 0, 0, 0, 0, 0]
    assert summary['joint_argmax_errors'] == 1


def test_projection_does_not_reselect_threshold_to_hide_error(logits):
    rows, z, spec = logits
    summary, _ = summarize_fixed(rows, z, dict(spec, threshold=.99))
    assert summary['threshold'] == .99 and summary['binary_errors'] == 1
    rows[0]['partition'] = 'train'
    with pytest.raises(ValueError, match='development'):
        summarize_fixed(rows, z, spec)


@pytest.mark.parametrize('fault', ['shape', 'nan', 'error', 'argmax', 'decision'])
def test_reload_requires_numerical_and_decision_parity(logits, fault):
    np = pytest.importorskip('numpy')
    _, reference, spec = logits; actual = reference.copy()
    if fault == 'shape': actual = actual[:, :2]
    elif fault == 'nan': actual[0, 0] = np.nan
    elif fault == 'error': actual[0, 0] = .01
    elif fault == 'argmax': actual[0, 1] = 1e-6
    else:
        reference[:] = 0; actual[:] = 0; actual[0, -1] = -1e-6; spec['threshold'] = .5
    with pytest.raises(ValueError):
        compare_logits(actual, reference, spec, limit=1e-5, exact_decisions=True)


def test_historical_drift_is_reported_and_not_relabeled_as_parity(logits):
    _, reference, spec = logits; actual = reference.copy(); actual[0, -1] = 4
    result = compare_logits(actual, reference, spec)
    assert result['maximum_logit_error'] == 4
    assert result['fixed_threshold_decision_differences'] == 1
    with pytest.raises(ValueError, match='tolerance'):
        compare_logits(actual, reference, spec, limit=.001)


def test_actual_head_projection_preserves_selected_row_order_and_reload():
    np = pytest.importorskip('numpy'); mx = pytest.importorskip('mlx.core')
    from training.shared_tail_training_v9 import make_head
    mx.set_default_device(mx.cpu); rng = np.random.default_rng(1090)
    x = rng.normal(size=(6, 12)).astype(np.float32); indices = np.array([4, 0, 3])
    weights = {'0.weight':rng.normal(size=(8, 12)).astype(np.float32)*.1,
        '0.bias':np.zeros(8, np.float32), '3.weight':rng.normal(size=(19, 8)).astype(np.float32)*.1,
        '3.bias':np.zeros(19, np.float32)}
    head = make_head(weights); head.eval()
    single = project_rows(head, x, indices, mx, np, 1)
    batch = project_rows(head, x, indices, mx, np, 256)
    direct = head(mx.array(x[indices])); mx.eval(direct)
    np.testing.assert_allclose(batch, direct, atol=1e-6, rtol=0)
    result = compare_logits(single, batch, {'mode':'binary', 'threshold':.5}, limit=1e-5, exact_decisions=True)
    assert result['argmax_rows_differing'] == 0 and single.shape == (3, 19)
