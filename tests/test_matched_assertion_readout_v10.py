"""Matched arrays, unchanged objective and an identical-input optimization control."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest

from training.matched_assertion_readout_v10 import paired_rows, arm_arrays, schedule


def metadata():
    old = [{'case_id': str(i), 'language': 'ro', 'partition': 'train' if i < 2 else 'dev',
            'gold': i % 2, 'fields': [0, 0, 2, 0, 0, i % 2], 'token_sha256': str(i)*64} for i in range(3)]
    added = [{'case_id': 'new'+str(j), 'pair_id': 'pair'+str(j), 'language': 'ro', 'partition': 'train',
              'gold': old[i]['gold'], 'fields': old[i]['fields'][:], 'parent_case_id': old[i]['case_id'],
              'parent_feature_index': i, 'parent_token_sha256': old[i]['token_sha256'],
              'token_sha256': str(j+5)*64} for j, i in enumerate((1, 0, 1))]
    return old, added


def test_pair_labels_and_original_dev_positions_are_preserved():
    original, additions = metadata(); rows, indices = paired_rows(original, additions)
    assert indices == [1, 0, 1] and rows[:len(original)] == original
    assert [r['gold'] for r in rows[3:]] == [1, 0, 1]
    assert [r['case_id'] for r in rows if r['partition'] == 'dev'] == ['2']
    assert schedule(rows)['training_rows'] == 5 and schedule(rows)['development_rows'] == 1


@pytest.mark.parametrize('fault', ['dev', 'index', 'gold', 'fields', 'token', 'identity', 'duplicate'])
def test_bad_parent_mapping_is_rejected(fault):
    original, additions = metadata()
    if fault == 'dev': original[1]['partition'] = 'dev'
    elif fault == 'index': additions[0]['parent_feature_index'] = 5
    elif fault == 'gold': additions[0]['gold'] = 0
    elif fault == 'fields': additions[0]['fields'][2] = 0
    elif fault == 'token': additions[0]['parent_token_sha256'] = 'bad'
    elif fault == 'identity': additions[0]['parent_case_id'] = 'bad'
    else: additions.append(deepcopy(additions[0]))
    with pytest.raises(ValueError): paired_rows(original, additions)


def test_only_appended_features_and_teacher_logits_differ_between_arms():
    np = pytest.importorskip('numpy')
    x = np.arange(12, dtype=np.float32).reshape(3, 4); z = np.arange(6, dtype=np.float32).reshape(3, 2)
    ax = np.ones((3, 4), np.float32)*100; az = np.ones((3, 2), np.float32)*200
    cx, cz = arm_arrays(x, z, ax, az, [1, 0, 1], 'matched-control')
    tx, tz = arm_arrays(x, z, ax, az, [1, 0, 1], 'wording-treatment')
    assert np.array_equal(cx[:3], tx[:3]) and np.array_equal(cz[:3], tz[:3])
    assert np.array_equal(cx[3:], x[[1, 0, 1]]) and np.array_equal(cz[3:], z[[1, 0, 1]])
    assert np.array_equal(tx[3:], ax) and np.array_equal(tz[3:], az)


@pytest.mark.parametrize('fault', ['arm', 'shape', 'dtype', 'nonfinite', 'index'])
def test_invalid_array_views_are_rejected(fault):
    np = pytest.importorskip('numpy')
    x = np.ones((3, 4), np.float32); z = np.ones((3, 2), np.float32)
    ax = x[:2].copy(); az = z[:2].copy(); indices = [0, 1]; arm = 'matched-control'
    if fault == 'arm': arm = 'other'
    elif fault == 'shape': ax = ax[:, :3]
    elif fault == 'dtype': ax = ax.astype(np.float64)
    elif fault == 'nonfinite': az[0, 0] = np.nan
    else: indices[0] = -1
    with pytest.raises(ValueError): arm_arrays(x, z, ax, az, indices, arm)


def loss_node(name):
    root = Path(__file__).resolve().parents[1]
    return next(n for n in ast.walk(ast.parse((root/'training'/name).read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == 'loss_fn')


def test_actual_training_objective_is_unchanged_from_verified75():
    assert ast.dump(loss_node('train_matched_assertion_readout_v10.py')) == ast.dump(loss_node('train_dropout_readout_v10.py'))


@pytest.mark.parametrize('dropout,coefficient', [(0., 1.), (.1, .25)])
def test_identical_inputs_produce_identical_training_updates_in_both_arms(dropout, coefficient):
    mx = pytest.importorskip('mlx.core'); np = pytest.importorskip('numpy')
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from training.native_fields_v9 import SIZES
    from training.dropout_readout_v10 import make_head
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass, policy_binary_cross_entropy
    mx.set_default_device(mx.cpu); rng = np.random.default_rng(79)
    x = rng.normal(size=(7, 12)).astype(np.float32); z = rng.normal(size=(7, 19)).astype(np.float32)
    indices = [2, 2, 4]; extra_x = x[indices].copy(); extra_z = z[indices].copy()
    initial = {'0.weight': rng.normal(size=(8, 12)).astype(np.float32)*.1,
               '0.bias': np.zeros(8, np.float32), '3.weight': rng.normal(size=(19, 8)).astype(np.float32)*.1,
               '3.bias': np.zeros(19, np.float32)}
    couplings = {'scope': np.ones((3, 3)).tolist(), 'assertion_relation': np.ones((4, 2)).tolist()}
    env = {'mx': mx, 'nn': nn, 'SIZES': SIZES, 'weights': [mx.ones(size) for size in SIZES],
           'coefficient': .2, 'policy_coefficient': coefficient,
           'policy_score': make_coupled_policy_mass(mx, couplings), 'policy_binary_cross_entropy': policy_binary_cross_entropy}
    node = loss_node('train_matched_assertion_readout_v10.py')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<actual79-objective>', 'exec'), env)
    fields = np.column_stack([rng.integers(size, size=10) for size in SIZES]).astype(np.int32)
    labels = rng.integers(2, size=10).astype(np.int32); results = []
    for arm in ('matched-control', 'wording-treatment'):
        features, teacher = arm_arrays(x, z, extra_x, extra_z, indices, arm)
        mx.random.seed(1011); order_rng = np.random.default_rng(1011)
        head = make_head(initial, dropout); mx.eval(head.parameters()); head.train()
        optimizer = optim.AdamW(learning_rate=.0001, weight_decay=.01)
        value_grad = nn.value_and_grad(head, env['loss_fn']); losses = []
        for _ in range(3):
            chosen = order_rng.permutation(10)[:4]
            loss, gradient = value_grad(head, mx.array(features[chosen]), mx.array(labels[chosen]),
                                        mx.array(fields[chosen]), mx.array(teacher[chosen]))
            gradient, norm = optim.clip_grad_norm(gradient, 1.); mx.eval(loss, norm, gradient)
            optimizer.update(head, gradient); mx.eval(head.parameters(), optimizer.state)
            losses.append(float(loss.item()))
        results.append((losses, {k: np.asarray(v).copy() for k, v in tree_flatten(head.parameters())}))
    assert results[0][0] == results[1][0]
    assert all(np.array_equal(results[0][1][k], value) for k, value in results[1][1].items())
