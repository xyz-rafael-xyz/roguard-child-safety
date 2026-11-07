import itertools
import json
from pathlib import Path

import pytest


def dependencies():
    np = pytest.importorskip('numpy')
    mx = pytest.importorskip('mlx.core')
    mx.set_default_device(mx.cpu)
    return np, mx


def coupling_tables(np, kind):
    from roguard.evidence import ONTOLOGY
    if kind in ('ro', 'uk'):
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / 'research/v10-evidence/crossed-review-readout-62'
                           / (kind + '-review.json')).read_text())['couplings']
    rng = np.random.default_rng(1071)
    return {name: rng.uniform(.1, 3., (len(ONTOLOGY[left]), len(ONTOLOGY[right]))).tolist()
            for name, left, right in (('scope', 'subject', 'source'),
                                     ('assertion_relation', 'assertion', 'relation'))}


def enumerated_mass(np, logits, couplings):
    """Independent reference: enumerate semantic worlds and apply the ontology."""
    from roguard.evidence import ONTOLOGY, disposition
    from training.native_fields_v9 import FIELDS, SIZES
    parts = np.split(np.asarray(logits, dtype=np.float64), np.cumsum(SIZES)[:-1], axis=-1)
    probabilities = []
    for part in parts:
        values = np.exp(part - part.max(axis=1, keepdims=True))
        probabilities.append(values / values.sum(axis=1, keepdims=True))
    total = np.zeros(len(logits))
    review = np.zeros(len(logits))
    for world in itertools.product(*(range(size) for size in SIZES)):
        weight = np.ones(len(logits))
        labels = {}
        for j, (field, value) in enumerate(zip(FIELDS, world)):
            weight *= probabilities[j][:, value]
            labels[field] = {'label': ONTOLOGY[field][value]}
        weight *= couplings['scope'][world[FIELDS.index('subject')]][world[FIELDS.index('source')]]
        weight *= couplings['assertion_relation'][world[FIELDS.index('assertion')]][world[FIELDS.index('relation')]]
        total += weight
        if disposition(labels)[0] != 'CONTINUE':
            review += weight
    return review / total


@pytest.mark.parametrize('kind', ['random', 'ro', 'uk'])
def test_policy_matches_world_enumeration_and_runtime(kind):
    np, mx = dependencies()
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass
    from training.native_fields_v9 import SIZES
    from roguard.gemma4_review import _score
    couplings = coupling_tables(np, kind)
    rng = np.random.default_rng(1071)
    logits = rng.normal(size=(12, sum(SIZES))).astype(np.float32)
    logits[-4:] *= 15  # Also check near-deterministic field marginals.
    score = make_coupled_policy_mass(mx, couplings)
    result = np.asarray(score(mx.array(logits)))
    reference = enumerated_mass(np, logits, couplings)
    runtime = np.array([_score(np.r_[row, [0., 0.]], {'mode': 'coupled_policy', 'couplings': couplings}) for row in logits])
    assert np.max(np.abs(result - reference)) < 2e-6
    assert np.max(np.abs(result - runtime)) < 2e-6


@pytest.mark.parametrize('kind', ['random', 'ro', 'uk'])
def test_loss_gradient_matches_finite_difference_and_torch(kind):
    np, mx = dependencies()
    torch = pytest.importorskip('torch')
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass, policy_binary_cross_entropy, LOSS_EPSILON
    from training.differentiable_policy_v9 import policy_mass as torch_mass
    from training.native_fields_v9 import FIELDS, SIZES
    couplings = coupling_tables(np, kind)
    logits = np.random.default_rng(2071).normal(size=(4, sum(SIZES))).astype(np.float32)
    labels = np.array([0., 1., 0., 1.], dtype=np.float32)
    mass = make_coupled_policy_mass(mx, couplings)
    def loss(z):
        return policy_binary_cross_entropy(mx, mass(z), mx.array(labels)).mean()
    loss_value, gradient = mx.value_and_grad(loss)(mx.array(logits))
    mx.eval(loss_value, gradient)
    actual = np.asarray(gradient)
    assert np.isfinite(actual).all()
    assert float(loss(mx.array(logits) - .1 * gradient).item()) < float(loss_value.item())
    z = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(labels, dtype=torch.float64)
    probability = torch_mass(torch, z, couplings).clamp(LOSS_EPSILON, 1 - LOSS_EPSILON)
    oracle = -(target * probability.log() + (1 - target) * torch.log1p(-probability)).mean()
    oracle.backward()
    assert abs(float(loss_value.item()) - oracle.item()) < 2e-6
    assert np.max(np.abs(actual - z.grad.numpy())) < 2e-6
    # Every semantic logit, using an independent enumerated-world calculation.
    def reference_loss(value):
        prob = np.clip(enumerated_mass(np, value, couplings), LOSS_EPSILON, 1 - LOSS_EPSILON)
        return -(labels * np.log(prob) + (1 - labels) * np.log1p(-prob)).mean()
    finite = np.zeros_like(logits, dtype=np.float64)
    step = 1e-3
    for index in np.ndindex(logits.shape):
        plus, minus = logits.astype(np.float64), logits.astype(np.float64)
        plus[index] += step
        minus[index] -= step
        finite[index] = (reference_loss(plus) - reference_loss(minus)) / (2 * step)
    assert np.max(np.abs(actual - finite)) < 2e-6
    for field, values in zip(FIELDS, np.split(actual, np.cumsum(SIZES)[:-1], axis=1)):
        if field == 'time':
            assert np.count_nonzero(values) == 0
        else:
            assert np.abs(values).sum() > 0


def test_bce_clamp_is_finite_at_endpoints():
    np, mx = dependencies()
    from training.coupled_policy_loss_v10 import policy_binary_cross_entropy
    probabilities = mx.array([0., 1., 0., 1.])
    labels = mx.array([0., 1., 1., 0.])
    values = policy_binary_cross_entropy(mx, probabilities, labels)
    gradient = mx.grad(lambda v: policy_binary_cross_entropy(mx, v, labels).sum())(probabilities)
    assert np.isfinite(np.asarray(values)).all()
    assert np.isfinite(np.asarray(gradient)).all()
    assert np.max(np.asarray(values)) < 14


@pytest.mark.parametrize('failure', ['missing', 'extra', 'shape', 'zero', 'negative', 'nan'])
def test_rejects_invalid_couplings(failure):
    np, mx = dependencies()
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass
    tables = coupling_tables(np, 'random')
    if failure == 'missing':
        del tables['scope']
    elif failure == 'extra':
        tables['other'] = [[1.]]
    elif failure == 'shape':
        tables['scope'] = [[1.]]
    else:
        tables['scope'][0][0] = {'zero': 0., 'negative': -1., 'nan': float('nan')}[failure]
    with pytest.raises(ValueError):
        make_coupled_policy_mass(mx, tables)


def test_rejects_invalid_input_shapes():
    np, mx = dependencies()
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass, policy_binary_cross_entropy
    from training.native_fields_v9 import SIZES
    mass = make_coupled_policy_mass(mx, coupling_tables(np, 'random'))
    for value in (mx.zeros((sum(SIZES),)), mx.zeros((2, sum(SIZES) + 2)), mx.zeros((0, sum(SIZES)))):
        with pytest.raises(ValueError):
            mass(value)
    with pytest.raises(ValueError):
        policy_binary_cross_entropy(mx, mx.zeros((2,)), mx.zeros((3,)))
