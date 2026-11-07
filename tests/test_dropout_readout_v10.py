"""Training-only dropout keeps exported inference and its zero-dropout control exact."""
import ast
from pathlib import Path

import pytest


def setup(probability):
    np = pytest.importorskip('numpy'); mx = pytest.importorskip('mlx.core')
    mx.set_default_device(mx.cpu)
    from training.native_fields_v9 import SIZES
    from training.shared_tail_training_v9 import make_head
    from training.dropout_readout_v10 import make_head as make_dropout_head
    rng = np.random.default_rng(1075)
    tensors = {'0.weight': rng.normal(size=(12, 16)).astype(np.float32),
               '0.bias': rng.normal(size=12).astype(np.float32),
               '3.weight': rng.normal(size=(sum(SIZES)+2, 12)).astype(np.float32),
               '3.bias': rng.normal(size=sum(SIZES)+2).astype(np.float32)}
    return np, mx, make_head(tensors), make_dropout_head(tensors, probability), mx.array(rng.normal(size=(64, 16)).astype(np.float32))


@pytest.mark.parametrize('probability', [0., .1, .3])
def test_exported_inventory_and_evaluation_are_identical(probability):
    np, mx, old, new, x = setup(probability)
    from mlx.utils import tree_flatten
    old.eval(); new.eval(); a, b = old(x), new(x); mx.eval(a, b)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    expected = dict(tree_flatten(old.parameters())); actual = dict(tree_flatten(new.parameters()))
    assert set(actual) == set(expected) == {'first.weight', 'first.bias', 'last.weight', 'last.bias'}
    for key in actual: assert np.array_equal(np.asarray(actual[key]), np.asarray(expected[key]))


@pytest.mark.parametrize('probability', [.1, .3])
def test_training_masks_are_seeded_and_disabled_for_evaluation(probability):
    np, mx, old, new, x = setup(probability)
    old.eval(); expected = old(x); mx.eval(expected)
    new.train(); mx.random.seed(1075); first = new(x); mx.eval(first)
    mx.random.seed(1075); repeat = new(x); mx.eval(repeat)
    mx.random.seed(1076); another = new(x); mx.eval(another)
    assert np.array_equal(np.asarray(first), np.asarray(repeat))
    assert not np.array_equal(np.asarray(first), np.asarray(another))
    assert not np.array_equal(np.asarray(first), np.asarray(expected))
    new.eval(); actual = new(x); mx.eval(actual)
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_actual_zero_dropout_objective_and_gradients_match_prior_trainer():
    np, mx, old, new, x = setup(0.)
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from training.native_fields_v9 import SIZES
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass, policy_binary_cross_entropy
    from roguard.evidence import ONTOLOGY
    rng = np.random.default_rng(1075)
    tables = {name: np.ones((len(ONTOLOGY[left]), len(ONTOLOGY[right]))).tolist()
              for name, left, right in (('scope', 'subject', 'source'), ('assertion_relation', 'assertion', 'relation'))}
    env = {'mx': mx, 'nn': nn, 'SIZES': SIZES, 'coefficient': .2, 'policy_coefficient': .25,
           'weights': [mx.ones(size) for size in SIZES], 'policy_score': make_coupled_policy_mass(mx, tables),
           'policy_binary_cross_entropy': policy_binary_cross_entropy}
    root = Path(__file__).resolve().parents[1]; losses = []
    for name in ('train_policy_consistent_readout_v10.py', 'train_dropout_readout_v10.py'):
        path = root/'training'/name
        nodes = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == 'loss_fn']
        assert len(nodes) == 1
        namespace = dict(env)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
        losses.append(namespace['loss_fn'])
    old.train(); new.train()
    gold = mx.array([0, 1]*32); fields = mx.array(np.column_stack([rng.integers(size, size=64) for size in SIZES]))
    teacher = mx.array(rng.normal(size=(64, sum(SIZES)+2)).astype(np.float32))
    a, ag = nn.value_and_grad(old, losses[0])(old, x, gold, fields, teacher)
    b, bg = nn.value_and_grad(new, losses[1])(new, x, gold, fields, teacher)
    mx.eval(a, b, ag, bg); assert float(a.item()) == float(b.item())
    expected = dict(tree_flatten(ag))
    for key, value in tree_flatten(bg): assert np.array_equal(np.asarray(value), np.asarray(expected[key]))


@pytest.mark.parametrize('probability', [-.1, 1., float('nan'), True])
def test_invalid_dropout_is_rejected(probability):
    from training.dropout_readout_v10 import make_head
    with pytest.raises(ValueError): make_head({}, probability)
