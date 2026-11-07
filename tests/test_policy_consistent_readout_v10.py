import ast
from pathlib import Path

import pytest


def isolated_loss(path, environment):
    """Run the actual nested loss without starting a training job."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'loss_fn']
    assert len(nodes) == 1
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = dict(environment)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['loss_fn']


@pytest.mark.parametrize('coefficient', [0., .25, 1.])
def test_actual_training_objective_preserves_control_and_adds_only_policy_term(coefficient):
    np = pytest.importorskip('numpy')
    mx = pytest.importorskip('mlx.core')
    nn = pytest.importorskip('mlx.nn')
    mx.set_default_device(mx.cpu)
    from mlx.utils import tree_flatten
    from roguard.evidence import ONTOLOGY
    from training.native_fields_v9 import SIZES
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass, policy_binary_cross_entropy
    rng = np.random.default_rng(3071)
    couplings = {name: rng.uniform(.1, 3., (len(ONTOLOGY[left]), len(ONTOLOGY[right]))).tolist()
                 for name, left, right in (('scope', 'subject', 'source'),
                                          ('assertion_relation', 'assertion', 'relation'))}
    score = make_coupled_policy_mass(mx, couplings)
    env = {'mx': mx, 'nn': nn, 'SIZES': SIZES, 'coefficient': .2,
           'weights': [mx.array(rng.uniform(.5, 2., size).astype(np.float32)) for size in SIZES],
           'policy_coefficient': coefficient, 'policy_score': score,
           'policy_binary_cross_entropy': policy_binary_cross_entropy}
    root = Path(__file__).resolve().parents[1]
    old = isolated_loss(root / 'training/train_semantic_readout_v10.py', env)
    new = isolated_loss(root / 'training/train_policy_consistent_readout_v10.py', env)
    mx.random.seed(3071)
    head = nn.Linear(5, sum(SIZES) + 2)
    features = mx.array(rng.normal(size=(8, 5)).astype(np.float32))
    gold = mx.array([0, 1] * 4)
    fields = mx.array(np.column_stack([rng.integers(size, size=8) for size in SIZES]).astype(np.int32))
    teacher = mx.array(rng.normal(size=(8, sum(SIZES) + 2)).astype(np.float32))
    args = (features, gold, fields, teacher)
    old_loss, old_grad = nn.value_and_grad(head, old)(head, *args)
    new_loss, new_grad = nn.value_and_grad(head, new)(head, *args)
    def policy_only(model):
        return policy_binary_cross_entropy(mx, score(model(features)[:, :sum(SIZES)]), gold).mean()
    policy_loss, policy_grad = nn.value_and_grad(head, policy_only)(head)
    mx.eval(old_loss, new_loss, policy_loss, old_grad, new_grad, policy_grad)
    assert abs(float(new_loss.item()) - (float(old_loss.item()) + coefficient * float(policy_loss.item()))) < 2e-6
    old_flat, new_flat, policy_flat = (dict(tree_flatten(g)) for g in (old_grad, new_grad, policy_grad))
    for key in old_flat:
        expected = np.asarray(old_flat[key]) + coefficient * np.asarray(policy_flat[key])
        assert np.max(np.abs(np.asarray(new_flat[key]) - expected)) < 2e-6
        if coefficient == 0:
            assert np.array_equal(np.asarray(new_flat[key]), np.asarray(old_flat[key]))
    if coefficient == 0:
        assert float(new_loss.item()) == float(old_loss.item())
