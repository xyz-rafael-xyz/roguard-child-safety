"""Exercise the actual training head, cached path, and completed-reference gates."""
import json
from types import SimpleNamespace

import pytest


def model_fixture():
    mx = pytest.importorskip('mlx.core'); np = pytest.importorskip('numpy')
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from roguard.gemma4_review import _make_model
    mx.set_default_device(mx.cpu); mx.random.seed(1091)
    base = Model(ModelArgs(hidden_size=16, num_hidden_layers=4, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=2, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=8,
        layer_types=['sliding_attention', 'full_attention']*2))
    model = _make_model(base, boundary=2, width=12,
        lora=dict(rank=2, scale=4., dropout=0., keys=['self_attn.q_proj', 'self_attn.o_proj']))
    return mx, np, model


@pytest.mark.parametrize('language,probability', [('ro', .1), ('uk', 0.)])
def test_dropout_head_preserves_export_and_inference_but_trains_tail(language, probability):
    mx, np, model = model_fixture()
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import make_prompt_cache
    from roguard.gemma4_attention import capture_with_prefix, replay
    from training.large_review_attention_v10 import check_trainable, make_optimizer
    from training.runtime_attention_continuation_v10 import install_head, native_cached_features
    weights = {k: np.asarray(v).copy() for k, v in tree_flatten(model.trainable_parameters())}
    x = mx.random.normal((64, 16)); original = model.heads[language](x); mx.eval(original)
    install_head(model, weights, language, probability)
    names = check_trainable(model, language, boundary=2)
    assert len(names) == 12
    assert all('dropout' not in name for name in dict(tree_flatten(model.parameters())))
    current = model.heads[language](x); mx.eval(current)
    assert np.array_equal(np.asarray(current), np.asarray(original))
    model.train(); active = model.heads[language](x); mx.eval(active)
    assert np.array_equal(np.asarray(active), np.asarray(original)) == (probability == 0.)
    model.eval(); inactive = model.heads[language](x); mx.eval(inactive)
    assert np.array_equal(np.asarray(inactive), np.asarray(original))
    cache = make_prompt_cache(model.base)
    mx.eval(model.base.model(mx.array([[1, 2]]), cache=cache))
    prefix = [(c.offset, [np.asarray(v).copy() for v in c.state]) for c in cache]
    # Short and window-crossing suffixes both exercise the actual native cache.
    for tokens in ([[3, 4, 5]], [[3, 4, 5, 6, 7, 8, 9, 10, 11]]):
        suffix = mx.array(tokens)
        state = capture_with_prefix(model.base, suffix, cache, start=2)
        hidden = replay(model.base, state)[:, 0, :]
        native = native_cached_features(model, suffix, cache); mx.eval(hidden, native)
        assert float(mx.max(mx.abs(hidden-native)).item()) < 2e-5
        for c, (offset, values) in zip(cache, prefix):
            assert c.offset == offset
            assert all(np.array_equal(np.asarray(v), old) for v, old in zip(c.state, values))
    frozen = {k: np.asarray(v).copy() for k, v in tree_flatten(model.parameters()) if k not in names}
    before = {k: np.asarray(v).copy() for k, v in tree_flatten(model.trainable_parameters())}
    model.train()
    def loss(m):
        return mx.square(m.heads[language](replay(m.base, state)[:, 0, :])).mean()
    value, gradients = nn.value_and_grad(model, loss)(model); mx.eval(value, gradients)
    optimizer, _ = make_optimizer(dict(learning_rate=1e-5, head_learning_rate=1e-4, weight_decay=.01), language)
    optimizer.update(model, gradients); mx.eval(model.parameters(), optimizer.state)
    for k, v in tree_flatten(model.parameters()):
        if k in frozen: assert np.array_equal(np.asarray(v), frozen[k]), k
    for scope in ('heads.', 'base.'):
        assert any(not np.array_equal(np.asarray(v), before[k])
                   for k, v in tree_flatten(model.trainable_parameters()) if k.startswith(scope))
    model.eval(); output = model.heads[language](replay(model.base, state)[:, 0, :]); mx.eval(output)
    _, _, deployed = model_fixture()
    exported = dict(weights)
    exported.update({k: np.asarray(v).copy() for k, v in tree_flatten(model.trainable_parameters())})
    from roguard.gemma4_review import _load_weights
    _load_weights(deployed, exported)
    # The deterministic fixture reconstructs the same frozen base.
    deployed_output = deployed.heads[language](native_cached_features(deployed, suffix, cache)); mx.eval(deployed_output)
    assert float(mx.max(mx.abs(output-deployed_output)).item()) < 2e-5


@pytest.mark.parametrize('fault', ['field', 'binary', 'nonfinite', 'magnitude'])
def test_reload_checks_every_head_and_served_decision(fault):
    np = pytest.importorskip('numpy')
    from training.runtime_attention_continuation_v10 import check_decisions
    saved = np.zeros((2, 19), dtype=np.float32); actual = saved.copy()
    if fault == 'field': actual[0, 7] = 1e-6
    elif fault == 'binary': actual[0, -2] = 1e-6
    elif fault == 'nonfinite': actual[0, 0] = np.nan
    else: actual[0, 0] = .01
    with pytest.raises(ValueError):
        check_decisions(actual, saved, dict(mode='binary', threshold=.5))


def prepared_fixture(tmp_path, monkeypatch):
    np = pytest.importorskip('numpy')
    from roguard.trajectory_data import file_sha, write_jsonl
    from training.prepare_runtime_warm_start_v10 import summarize_fixed, compare_logits
    from eval.verify_matched_assertion_readout_v10 import paired_counts
    root = tmp_path/'repo'; private = tmp_path/'private'; evidence = root/'research/v10-evidence'
    warm = evidence/'runtime-warm-start-90'; data = private/'runtime-warm-start-90'
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    source = root/'training/prepare_runtime_warm_start_v10.py'; write(source, 'fixture')
    write(warm/'source.py', 'fixture'); write(evidence/'protocol-amendment-74.json', {})
    monkeypatch.setattr('training.prepare_runtime_warm_start_v10.load', lambda *a, **k: ({}, {'verified': True}))
    flags = dict(encoder_loaded=False, training_performed=False, held_out_test_opened=False, qualified=False)
    protocol = dict(flags, source_sha256=file_sha(source), plan_sha256=file_sha(evidence/'protocol-amendment-74.json'),
                    bindings={'verified': True}, initialization='selected75')
    write(warm/'protocol.json', protocol)
    queue_path = private/'runtime-warm-start-sequence-90-plan.json'
    queue = dict(source_hashes={'training/prepare_runtime_warm_start_v10.py': file_sha(source)},
                 steps=[dict(name='prepare')])
    write(queue_path, queue)
    write(evidence/'runtime-warm-start-sequence-90/report.json',
          dict(execution_complete=True, phase='finished', plan_sha256=file_sha(queue_path),
               source_hashes=queue['source_hashes'], completed_steps=[dict(name='prepare', exit_code=0)]))
    report = dict(flags, complete=True, decision='WARM_START_REFERENCE_ONLY_NO_MODEL_QUALIFICATION', languages={})
    for lang in ('ro', 'uk'):
        spec = dict(mode='binary', threshold=.5)
        rows = [dict(case_id=str(i), language=lang, partition='dev', gold=i % 2, fields=[0]*6) for i in range(4)]
        z = np.zeros((4, 19), np.float32); z[:, -2:] = [[1, -1], [-1, 1]]*2
        for directory in (data, private/'dropout-readout-75', private/'runtime-review-baseline-88'):
            directory.mkdir(parents=True, exist_ok=True)
        for directory in (data, private/'dropout-readout-75'):
            (directory/(lang+'-selected.safetensors')).write_bytes(b'fixture-weights')
            np.save(directory/(lang+'-selected-dev-logits.npy'),
                    z.astype(np.float64) if directory.name == 'dropout-readout-75' else z)
        np.save(data/(lang+'-cpu-dev-logits.npy'), z)
        np.save(private/'runtime-review-baseline-88'/(lang+'-warm-start-dev-logits.npy'), z)
        write_jsonl(data/(lang+'-dev-rows.jsonl'), rows)
        write_jsonl(private/'runtime-review-baseline-88'/(lang+'-rows.jsonl'), rows)
        for directory in (warm, evidence/'dropout-readout-75', evidence/'policy-consistent-readout-71'):
            write(directory/(lang+'-review.json'), spec)
        summary, decisions = summarize_fixed(rows, z, spec)
        comparison = compare_logits(z, z, spec)
        pairs = paired_counts([r['gold'] for r in rows], decisions, decisions)
        report['languages'][lang] = dict(runtime75=summary, original75=summary, runtime71=summary,
            cpu_replay=comparison, historical_logit_difference=comparison, gpu_reload=comparison,
            runtime75_vs_original75=pairs, runtime75_vs_runtime71=pairs,
            artifacts={p.name: file_sha(p) for p in data.glob(lang+'-*')})
    write(warm/'report.json', report)
    return SimpleNamespace(root=root, private=private, report=report, report_path=warm/'report.json', data=data)


def test_completed_reference_recounts_saved_metrics_and_binds_all_artifacts(tmp_path, monkeypatch):
    from training.runtime_attention_continuation_v10 import runtime_inputs
    f = prepared_fixture(tmp_path, monkeypatch)
    result = runtime_inputs(f.root, f.private)
    assert len(result['warm_artifacts']) == 8
    assert set(result['warm_specifications']) == {'ro', 'uk'}


def test_historical_float64_reproduces_old_guard_failure_without_rewriting_inputs(tmp_path, monkeypatch):
    from training.runtime_attention_v10 import runtime_inputs as previous
    from training.runtime_attention_continuation_v10 import runtime_inputs
    from roguard.trajectory_data import file_sha
    f = prepared_fixture(tmp_path, monkeypatch)
    paths = list(f.private.rglob('*.npy'))
    before = {str(p): file_sha(p) for p in paths}
    with pytest.raises(ValueError, match='Invalid runtime initialization logits'):
        previous(f.root, f.private)
    runtime_inputs(f.root, f.private)
    assert {str(p): file_sha(p) for p in paths} == before


@pytest.mark.parametrize('artifact', ['gpu', 'cpu', 'historical75', 'initial71'])
@pytest.mark.parametrize('fault', ['dtype', 'shape', 'nonfinite'])
def test_each_reference_has_its_own_strict_dtype_shape_and_finiteness(tmp_path, monkeypatch, artifact, fault):
    np = pytest.importorskip('numpy')
    from training.runtime_attention_continuation_v10 import runtime_inputs
    from roguard.trajectory_data import file_sha
    f = prepared_fixture(tmp_path, monkeypatch)
    path = {
        'gpu': f.data/'ro-selected-dev-logits.npy',
        'cpu': f.data/'ro-cpu-dev-logits.npy',
        'historical75': f.private/'dropout-readout-75/ro-selected-dev-logits.npy',
        'initial71': f.private/'runtime-review-baseline-88/ro-warm-start-dev-logits.npy',
    }[artifact]
    values = np.load(path, allow_pickle=False)
    if fault == 'dtype': values = values.astype(np.float32 if artifact == 'historical75' else np.float64)
    elif fault == 'shape': values = values[:-1]
    else: values[0, 0] = np.nan
    np.save(path, values)
    if artifact in ('gpu', 'cpu'):
        f.report['languages']['ro']['artifacts'][path.name] = file_sha(path)
        f.report_path.write_text(json.dumps(f.report))
    with pytest.raises(ValueError, match='Invalid runtime initialization logits: '+artifact):
        runtime_inputs(f.root, f.private)


@pytest.mark.parametrize('fault', ['incomplete', 'scope', 'metrics', 'pairs', 'cpu', 'reload', 'inventory', 'bytes', 'spec', 'sequence'])
def test_completed_reference_corruption_rejected(tmp_path, monkeypatch, fault):
    from training.runtime_attention_continuation_v10 import runtime_inputs
    f = prepared_fixture(tmp_path, monkeypatch)
    if fault == 'incomplete': f.report['complete'] = False
    elif fault == 'scope': f.report['training_performed'] = True
    elif fault == 'metrics': f.report['languages']['ro']['runtime75'] = dict(f.report['languages']['ro']['runtime75'], binary_errors=1)
    elif fault == 'pairs': f.report['languages']['ro']['runtime75_vs_original75'] = {}
    elif fault == 'cpu': f.report['languages']['ro']['cpu_replay'] = {}
    elif fault == 'reload': f.report['languages']['ro']['gpu_reload'] = dict(maximum_logit_error=.002, argmax_rows_differing=0, fixed_threshold_decision_differences=0)
    elif fault == 'inventory': f.report['languages']['ro']['artifacts'].pop('ro-cpu-dev-logits.npy')
    elif fault == 'bytes': (f.data/'ro-selected.safetensors').write_bytes(b'tampered')
    elif fault == 'spec': (f.report_path.parent/'ro-review.json').write_text('{}')
    else: (f.report_path.parent.parent/'runtime-warm-start-sequence-90/report.json').write_text('{"execution_complete": false}')
    f.report_path.write_text(json.dumps(f.report))
    with pytest.raises(ValueError): runtime_inputs(f.root, f.private)


def test_full_training_loop_on_small_real_42_layer_model(tmp_path, monkeypatch):
    """Run both languages, three passes, checkpoint selection and native reload.

    The tiny base and input loader substitute only fixture data; optimizer,
    loss, capture/replay, dropout, exports and serving-head math are real.
    """
    mx = pytest.importorskip('mlx.core'); np = pytest.importorskip('numpy')
    import sys
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from safetensors.numpy import save_file, load_file
    from roguard.gemma4_review import _make_model
    from roguard.gemma4_attention import capture_with_prefix, replay
    from roguard.evidence import ONTOLOGY
    from roguard.trajectory_data import file_sha
    import training.train_runtime_review_attention_continuation_v10 as trainer
    mx.set_default_device(mx.cpu)
    def make_model():
        mx.random.seed(7)
        base = Model(ModelArgs(hidden_size=16, num_hidden_layers=42, intermediate_size=32,
            num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
            num_kv_shared_layers=18, vocab_size=64, vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=8, sliding_window=8,
            layer_types=['sliding_attention', 'full_attention']*21))
        return _make_model(base, boundary=24, width=12)
    model = make_model()
    tensors = {k: np.asarray(v).copy() for k, v in tree_flatten(model.trainable_parameters())}
    cache = make_prompt_cache(model.base)
    mx.eval(model.base.model(mx.array([[1, 2]]), cache=cache))
    root = tmp_path/'root'; private = tmp_path/'private'; study = root/'study'; output = private/'fit'
    candidate = root/'v9'; initial_study = root/'initial'; initial_data = private/'initial'
    baseline_data = private/'runtime-review-baseline-88'; warm_data = private/'runtime-warm-start-90'
    for p in (candidate/'actual', initial_study, initial_data, baseline_data, warm_data): p.mkdir(parents=True)
    couplings = {name: np.ones((len(ONTOLOGY[left]), len(ONTOLOGY[right]))).tolist()
                 for name, left, right in (('scope', 'subject', 'source'), ('assertion_relation', 'assertion', 'relation'))}
    specs = {}; all_rows = {}
    for lang in ('ro', 'uk'):
        specs[lang] = dict(mode='coupled_policy', threshold=.5, couplings=couplings)
        (initial_study/(lang+'-review.json')).write_text(json.dumps(specs[lang]))
        for p in (candidate/'actual', initial_data): save_file(tensors, str(p/(lang+'-selected.safetensors')))
        rows = []; features = []; logits = []
        for i in range(4):
            tokens = [1, 2, 3+i, 9+i]
            rows.append(dict(case_id=lang+str(i), language=lang, group='fixture', partition='train' if i < 2 else 'dev',
                             fields=[i % 2]*6, gold=i % 2, tokens=tokens, payload={'tokens': tokens}))
            state = capture_with_prefix(model.base, mx.array([tokens[2:]]), cache, start=24)
            hidden = replay(model.base, state)[:, 0, :]; z = model.heads[lang](hidden); mx.eval(hidden, z)
            features.append(np.asarray(hidden)[0].copy()); logits.append(np.asarray(z)[0].copy())
        all_rows[lang] = rows
        np.save(baseline_data/(lang+'-features.npy'), np.stack(features))
        np.save(baseline_data/(lang+'-logits.npy'), np.stack(logits))
        np.save(warm_data/(lang+'-selected-dev-logits.npy'), np.stack(logits)[2:])
    class Engine:
        def __init__(self, *args):
            self.model = make_model(); self.make_cache = make_prompt_cache; self.tokenizer = None
    class Frozen:
        def __init__(self, *args):
            self.models = {lang: SimpleNamespace(_spec=specs[lang], _args=()) for lang in specs}
    monkeypatch.setattr('roguard.gemma4_review._Engine', Engine)
    monkeypatch.setattr('eval.run_trajectory_v10.LocalFrozenV9', Frozen)
    monkeypatch.setattr('roguard.gemma4_contract.PREFIX_TOKENS', [1, 2])
    monkeypatch.setattr('roguard.gemma4_contract.classifier_tokens', lambda tokenizer, payload: payload['tokens'])
    monkeypatch.setattr(trainer, 'load_inputs', lambda args: (all_rows, {'fixture': True}))
    sequence = root/'research/v10-evidence/runtime-warm-start-sequence-90/report.json'
    sequence.parent.mkdir(parents=True)
    sequence.write_text(json.dumps(dict(execution_complete=True, phase='finished', completed_steps=[{'exit_code': 0}])))
    plan = root/'plan.json'
    plan.write_text(json.dumps(dict(source_sha256=file_sha(trainer.__file__), input_bindings={'fixture': True},
        optimization=dict(epochs=3, batch_size=8, seed=1073, learning_rate=1e-5, head_learning_rate=1e-4,
                          weight_decay=.01, gradient_norm_clip=1., teacher_kl=.2, policy_ce={'ro': .25, 'uk': 1.},
                          head_dropout={'ro': .1, 'uk': 0.}),
        schedules={lang: dict(updates_per_epoch=1, total_updates=3, total_presentations=6) for lang in specs},
        populations={lang: dict(train=2, dev=2) for lang in specs}, dependency_pid=2**30)))
    args = dict(base=root/'base', candidate=candidate, input_study=root/'unused', inputs=root/'unused',
                feature_study=root/'unused', features=root/'unused', initial_study=initial_study,
                initial_private=initial_data, study=study, private=output, amendment=plan,
                root=root, private_root=private)
    monkeypatch.setattr(sys, 'argv', ['trainer']+[part for name, value in args.items()
                        for part in ('--'+name.replace('_', '-'), str(value))])
    trainer.main()
    report = json.loads((study/'report.json').read_text())
    assert report['complete'] and report['held_out_test_opened'] is False and report['stable_release_ready'] is False
    for lang, record in report['languages'].items():
        assert record['complete'] and record['trainable_tensors'] == 76
        assert [c['epoch'] for c in record['checkpoints']] == [0, 1, 2, 3]
        assert [c['completed_training_updates'] for c in record['checkpoints']] == [0, 1, 2, 3]
        assert [c['completed_training_presentations'] for c in record['checkpoints']] == [0, 2, 4, 6]
        assert record['selected_reload_decision_parity']['argmax_rows_differing'] == 0
        zero = load_file(output/lang/'epoch-0.safetensors')
        final = load_file(output/lang/'epoch-3.safetensors')
        assert all(np.array_equal(v, tensors[k]) for k, v in zero.items())
        assert any(not np.array_equal(v, tensors[k]) for k, v in final.items() if k.startswith('base.'))
        assert any(not np.array_equal(v, tensors[k]) for k, v in final.items() if k.startswith('heads.'+lang+'.'))
        other = 'uk' if lang == 'ro' else 'ro'
        assert all(np.array_equal(v, tensors[k]) for k, v in final.items() if k.startswith('heads.'+other+'.'))
