"""Streaming binary fit: source alignment, real tail gradients and unchanged loss."""
from copy import deepcopy
from pathlib import Path

import pytest

from roguard.assurance import digest
from roguard.gemma4_contract import PREFIX_TOKENS
from training.large_review_attention_v10 import training_schedule, validate_rows


def inputs():
    metadata, recovered = {}, []
    for lang in ('ro', 'uk'):
        metadata[lang] = []
        for i, split in enumerate(('train', 'train', 'dev')):
            tokens = PREFIX_TOKENS+[100+i+(10 if lang == 'uk' else 0)]
            row = {'case_id': f'{lang}-{i}', 'language': lang, 'partition': split, 'group': 'fixture',
                   'gold': i % 2, 'fields': [0]*6, 'token_sha256': digest(tokens)}
            metadata[lang].append(row)
            recovered.append({**row, 'feature_index': i, 'tokens': tokens, 'token_count': len(tokens),
                              'payload': {'language': lang, 'turns': [{'role': 'minor', 'text': str(i)}]},
                              'source_name': 'fixture', 'source_row_sha256': '0'*64, 'source_focus': 0})
    return recovered, metadata


def test_recovered_input_order_cannot_change_teacher_alignment():
    rows, metadata = inputs()
    result = validate_rows(list(reversed(rows)), metadata)
    assert [r['case_id'] for r in result['ro']] == ['ro-0', 'ro-1', 'ro-2']
    assert training_schedule(result['ro'], 3, 8) == {
        'updates_per_epoch': 1, 'total_updates': 3, 'total_presentations': 6}


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'position', 'label', 'tokens', 'prefix', 'partition', 'overlap'])
def test_recovered_inputs_fail_closed(fault):
    rows, metadata = inputs()
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows.append(deepcopy(rows[0]))
    elif fault == 'position':
        rows[0]['feature_index'] = 1
    elif fault == 'label':
        rows[0]['fields'] = [1]*6
    elif fault in ('tokens', 'prefix'):
        rows[0]['tokens'][-1 if fault == 'tokens' else 0] += 1
        if fault == 'prefix':
            rows[0]['token_sha256'] = metadata['ro'][0]['token_sha256'] = digest(rows[0]['tokens'])
    elif fault == 'partition':
        rows[0]['partition'] = metadata['ro'][0]['partition'] = 'test'
    elif fault == 'overlap':
        rows[2]['tokens'] = rows[0]['tokens'][:]
        rows[2]['token_count'] = len(rows[2]['tokens'])
        rows[2]['token_sha256'] = metadata['ro'][2]['token_sha256'] = rows[0]['token_sha256']
    with pytest.raises(ValueError):
        validate_rows(rows, metadata)


def numerical_setup():
    mx = pytest.importorskip('mlx.core'); np = pytest.importorskip('numpy')
    import mlx.nn as nn
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from roguard.gemma4_review import _make_model
    mx.set_default_device(mx.cpu); mx.random.seed(1073)
    base = Model(ModelArgs(hidden_size=16, num_hidden_layers=4, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=2, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=8,
        layer_types=['sliding_attention', 'full_attention']*2))
    model = _make_model(base, boundary=2, width=12,
        lora=dict(rank=2, scale=4., dropout=0., keys=['self_attn.q_proj', 'self_attn.o_proj']))
    for layer in model.base.model.layers[2:]:
        for projection in ('q_proj', 'o_proj'):
            adapter = getattr(layer.self_attn, projection)
            adapter.lora_b = .01*mx.random.normal(adapter.lora_b.shape)
    model.heads['uk'].freeze()
    from roguard.evidence import ONTOLOGY
    rng = np.random.default_rng(1073)
    couplings = {name: rng.uniform(.5, 2., (len(ONTOLOGY[left]), len(ONTOLOGY[right]))).tolist()
                 for name, left, right in (('scope', 'subject', 'source'), ('assertion_relation', 'assertion', 'relation'))}
    return mx, np, nn, model, couplings


@pytest.mark.parametrize('coefficient', [.25, 1.])
def test_streaming_objective_exactly_matches_actual_prior_head_training(coefficient):
    mx, np, nn, model, couplings = numerical_setup()
    import ast
    from mlx.utils import tree_flatten
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass, policy_binary_cross_entropy
    from training.large_review_attention_v10 import field_weights, make_loss
    from training.native_fields_v9 import SIZES
    rng = np.random.default_rng(1073)
    fields = np.column_stack([rng.integers(size, size=8) for size in SIZES]).astype(np.int32)
    weights = [mx.array(v) for v in field_weights(fields)]
    path = Path(__file__).resolve().parents[1]/'training/train_policy_consistent_readout_v10.py'
    nodes = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == 'loss_fn']
    assert len(nodes) == 1
    env = {'mx': mx, 'nn': nn, 'SIZES': SIZES, 'weights': weights, 'coefficient': .2,
           'policy_coefficient': coefficient, 'policy_score': make_coupled_policy_mass(mx, couplings),
           'policy_binary_cross_entropy': policy_binary_cross_entropy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), env)
    objective = make_loss(mx, nn, couplings, weights, teacher_kl=.2, policy_ce=coefficient)
    head = model.heads['ro']
    features = mx.array(rng.normal(size=(8, 16)).astype(np.float32)); gold = mx.array([0, 1]*4)
    fields = mx.array(fields); teacher = mx.array(rng.normal(size=(8, sum(SIZES)+2)).astype(np.float32))
    old_value, old_grad = nn.value_and_grad(head, env['loss_fn'])(head, features, gold, fields, teacher)
    def new(m):
        return objective(m(features), gold, fields, teacher)
    value, grad = nn.value_and_grad(head, new)(head)
    mx.eval(value, old_value, grad, old_grad)
    assert float(value.item()) == float(old_value.item())
    expected = dict(tree_flatten(old_grad))
    for name, v in tree_flatten(grad):
        assert np.array_equal(np.asarray(v), np.asarray(expected[name])), name


def test_prefix_replay_policy_gradients_and_optimizer_preserve_frozen_parameters():
    mx, np, nn, model, couplings = numerical_setup()
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import make_prompt_cache
    from roguard.gemma4_attention import capture_with_prefix, replay
    from training.large_review_attention_v10 import check_trainable, make_loss, make_optimizer
    from training.native_fields_v9 import SIZES
    names = check_trainable(model, 'ro', boundary=2)
    assert len(names) == 12
    tokens = [mx.array([[1, 2, 3, 4, 5]]), mx.array([[1, 2, 8, 9, 10, 11, 12, 13, 14]])]
    cache = make_prompt_cache(model.base); mx.eval(model.base.model(mx.array([[1, 2]]), cache=cache))
    states = [capture_with_prefix(model.base, t[:, 2:], cache, start=2) for t in tokens]
    gold = mx.array([0, 1]); fields = mx.array([[0]*6, [1]*6])
    objective = make_loss(mx, nn, couplings, [mx.ones((size,)) for size in SIZES], teacher_kl=.2, policy_ce=1.)
    teacher = mx.zeros((2, sum(SIZES)+2))
    def forward(m, cached):
        h = [replay(m.base, s)[:, 0, :] for s in states] if cached else [m.base.model(t)[:, -1, :] for t in tokens]
        return mx.concatenate([m.heads['ro'](v) for v in h])
    def loss(m, cached):
        return objective(forward(m, cached), gold, fields, teacher)
    a, ag = nn.value_and_grad(model, loss)(model, True)
    b, bg = nn.value_and_grad(model, loss)(model, False)
    mx.eval(a, ag, b, bg)
    assert abs(float(a.item())-float(b.item())) < 2e-5
    expected = dict(tree_flatten(bg))
    for name, v in tree_flatten(ag):
        assert float(mx.max(mx.abs(v-expected[name])).item()) < 3e-4, name
    frozen = {k: np.asarray(v).copy() for k, v in tree_flatten(model.parameters()) if k not in names}
    old = {k: np.asarray(v).copy() for k, v in tree_flatten(model.trainable_parameters())}
    config = {'learning_rate': 1e-5, 'head_learning_rate': 1e-4, 'weight_decay': .01}
    optimizer, _ = make_optimizer(config, 'ro')
    optimizer.update(model, ag); mx.eval(model.parameters(), optimizer.state)
    for k, v in tree_flatten(model.parameters()):
        if k in frozen:
            assert np.array_equal(np.asarray(v), frozen[k]), k
    assert any(not np.array_equal(np.asarray(v), old[k]) for k, v in tree_flatten(model.trainable_parameters()) if k.startswith('base.'))
    assert any(not np.array_equal(np.asarray(v), old[k]) for k, v in tree_flatten(model.trainable_parameters()) if k.startswith('heads.'))
    cached, full = forward(model, True), forward(model, False); mx.eval(cached, full)
    assert float(mx.max(mx.abs(cached-full)).item()) < 2e-5
    with pytest.raises(ValueError):
        model.heads['uk'].unfreeze(); check_trainable(model, 'ro', boundary=2)


def test_development_selection_retains_fixed_v9_comparison():
    np = pytest.importorskip('numpy')
    from training.large_review_attention_v10 import summarize
    from training.native_fields_v9 import SIZES
    from roguard.evidence import ONTOLOGY
    couplings = {name: np.ones((len(ONTOLOGY[left]), len(ONTOLOGY[right]))).tolist()
                 for name, left, right in (('scope', 'subject', 'source'), ('assertion_relation', 'assertion', 'relation'))}
    rows = [{'case_id': str(i), 'language': 'ro', 'group': str(i // 2), 'gold': i % 2, 'fields': [0]*6} for i in range(4)]
    logits = np.zeros((4, sum(SIZES)+2))
    logits[:, -2:] = [[10, -10], [-10, 10]]*2
    summary, predictions = summarize(rows, logits, couplings, [0, 0, 1, 1])
    assert summary['binary_errors'] == summary['joint_argmax_errors'] == summary['introduced_v9'] == 0
    assert summary['fixed_v9'] == 2
    assert [p['baseline_v9'] for p in predictions] == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        summarize(rows, logits, couplings, [0, 0, 1])


def test_binary_optimizer_routes_head_and_attention_to_registered_rates():
    mx, np, _, model, _ = numerical_setup()
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from training.large_review_attention_v10 import make_optimizer
    config = {'learning_rate': 1e-5, 'head_learning_rate': 1e-4, 'weight_decay': .01}
    combined, groups = make_optimizer(config, 'ro')
    separate = {name: optim.AdamW(learning_rate=rate, weight_decay=.01)
                for name, rate in (('heads', 1e-4), ('base', 1e-5))}
    expected = model.trainable_parameters()
    for scale in (.5, 1.):
        gradients = tree_map(lambda value: mx.ones_like(value), model.trainable_parameters())
        for optimizer, rate in groups:
            optimizer.learning_rate = rate*scale
        next_expected = {}
        for name, rate in (('heads', 1e-4), ('base', 1e-5)):
            separate[name].learning_rate = rate*scale
            next_expected.update(separate[name].apply_gradients({name: gradients[name]}, expected))
        combined.update(model, gradients); expected = next_expected
        mx.eval(model.trainable_parameters(), expected, combined.state)
        target = dict(tree_flatten(expected))
        for name, value in tree_flatten(model.trainable_parameters()):
            assert np.array_equal(np.asarray(value), np.asarray(target[name])), name
