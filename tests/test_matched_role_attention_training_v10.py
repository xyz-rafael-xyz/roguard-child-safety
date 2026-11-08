"""Exercise the paired driver with actual MLX optimization on a tiny Gemma4."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('arm', ['control', 'treatment'])
def test_driver_trains_both_languages_and_reloads_selected_checkpoint(tmp_path, monkeypatch, arm):
    mx = pytest.importorskip('mlx.core'); np = pytest.importorskip('numpy')
    pytest.importorskip('mlx_lm')
    from mlx.utils import tree_flatten
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from mlx_lm.models.cache import make_prompt_cache
    from safetensors.numpy import load_file, save_file
    from roguard.assurance import digest
    from roguard.trajectory_data import file_sha
    from roguard.gemma4_review import _make_model, _load_weights
    from roguard.gemma4_attention import capture_with_prefix, replay
    from training.matched_role_diversity_v10 import matched_population
    from training.matched_role_attention_inputs_v10 import OPTIMIZATION
    from training.large_review_attention_v10 import check_trainable
    from training.runtime_attention_continuation_v10 import install_head
    from training import train_matched_role_attention_v10 as trainer
    mx.set_default_device(mx.cpu)
    prefix_tokens = [1, 2]
    monkeypatch.setattr('roguard.gemma4_contract.PREFIX_TOKENS', prefix_tokens)
    monkeypatch.setattr('roguard.gemma4_contract.classifier_tokens', lambda tokenizer, payload: payload['fixture_tokens'])
    def build():
        mx.random.seed(1927)
        base = Model(ModelArgs(hidden_size=8, num_hidden_layers=42, intermediate_size=16,
            num_attention_heads=2, head_dim=4, global_head_dim=4, num_key_value_heads=1,
            num_kv_shared_layers=18, vocab_size=32, vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=4, sliding_window=8,
            layer_types=['sliding_attention', 'full_attention']*21))
        model = _make_model(base, boundary=24, width=12,
            lora=dict(rank=2, scale=4., dropout=0., keys=['self_attn.q_proj', 'self_attn.o_proj']))
        model.eval(); mx.eval(model.parameters())
        return model
    def engine(base_path, weights, expected):
        assert file_sha(weights) == expected
        model = build(); _load_weights(model, load_file(weights))
        return SimpleNamespace(model=model, make_cache=make_prompt_cache, tokenizer=None)
    monkeypatch.setattr('roguard.gemma4_review._Engine', engine)
    def project(model, rows, lang):
        prefix = make_prompt_cache(model.base)
        mx.eval(model.base.model(mx.array([prefix_tokens]), cache=prefix))
        x, z = [], []
        for row in rows:
            state = capture_with_prefix(model.base, mx.array([row['tokens'][2:]]), prefix, start=24)
            hidden = replay(model.base, state)[:, 0, :]
            logits = model.heads[lang](hidden); mx.eval(hidden, logits)
            x.append(np.asarray(hidden)[0].copy()); z.append(np.asarray(logits)[0].copy())
        return np.stack(x), np.stack(z)
    root = Path(trainer.__file__).resolve().parents[1]
    private = tmp_path/'private'; private.mkdir()
    data, inventory = {}, {}
    for lang in ('ro', 'uk'):
        def row(i, split='train', new=False):
            tokens = prefix_tokens+[i+3]
            return dict(case_id=f'{lang}-{i}', language=lang, partition=split, group='fixture',
                tokens=tokens, token_count=len(tokens), token_sha256=digest(tokens),
                fields=[0, 0, 0, 0, 0, i % 2], gold=i % 2,
                payload={'fixture_tokens': tokens, 'language': lang}, synthetic_only=new, independently_annotated=False)
        base = [row(0), row(1), row(2, 'dev'), row(3, 'dev')]
        new = [row(4, new=True), row(5, new=True)]
        paired = matched_population(base, new, lang)
        model = build()
        original = {k: np.asarray(v).copy() for k, v in tree_flatten(model.trainable_parameters())}
        initial = {k: v.copy() for k, v in original.items()}
        initial[f'heads.{lang}.first.bias'] += .01
        initial['base.model.layers.24.self_attn.q_proj.lora_b'] += .005
        original_path = private/(lang+'-original.safetensors'); initial_path = private/(lang+'-initial.safetensors')
        save_file(original, str(original_path)); save_file(initial, str(initial_path))
        _, base_teacher = project(model, base, lang); _, new_teacher = project(model, new, lang)
        _load_weights(model, initial); install_head(model, initial, lang, OPTIMIZATION['head_dropout'][lang])
        inventory[lang] = sorted(check_trainable(model, lang))
        x, z = project(model, new, lang); _, initial_dev = project(model, base[2:], lang)
        spec = dict(mode='binary', threshold=.5, couplings=dict(scope=[[1.]*3]*3, assertion_relation=[[1.]*2]*4))
        data[lang] = dict(**paired, base=base, new=new, base_teacher=base_teacher, new_teacher=new_teacher,
            new_initial_features=x, new_initial_logits=z, initial_dev_logits=initial_dev,
            weights={'v9': original_path, 'selected95': initial_path},
            specs={'v9': dict(spec, adapter_sha256=file_sha(original_path)),
                   'selected95': dict(spec, adapter_sha256=file_sha(initial_path))})
    schedules = {lang: dict(updates_per_epoch=2, total_updates=2, total_presentations=4) for lang in data}
    binding = dict(schedules=schedules, trainable_inventory=inventory)
    plan = dict(optimization=dict(OPTIMIZATION, epochs=1, batch_size=2),
                populations={lang: dict(train=4, dev=2) for lang in data}, source_sha256=file_sha(trainer.__file__))
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    study, output = tmp_path/'study', private/'output'
    monkeypatch.setattr(trainer, 'load_inputs', lambda args: (data, binding, plan))
    monkeypatch.setattr('sys.argv', ['train', '--base', str(tmp_path/'base'), '--study', str(study),
        '--private', str(output), '--plan', str(plan_path), '--plan-sha256', file_sha(plan_path),
        '--root', str(root), '--private-root', str(private), '--arm', arm])
    trainer.main()
    report = json.loads((study/'report.json').read_text())
    assert report['complete'] and report['arm'] == arm and report['held_out_test_opened'] is False
    for lang, record in report['languages'].items():
        assert record['complete'] and len(record['checkpoints']) == 2
        assert record['checkpoints'][-1]['completed_training_updates'] == 2
        assert record['checkpoints'][-1]['completed_training_presentations'] == 4
        assert record['initial_new_parity']['maximum_logit_error'] <= 1e-5
        assert record['selected_reload_decision_parity']['argmax_rows_differing'] == 0
        assert record['selected_reload_decision_parity']['fixed_threshold_decision_differences'] == 0
        initial = load_file(data[lang]['weights']['selected95'])
        zero = load_file(output/lang/'epoch-0.safetensors')
        trained = load_file(output/lang/'epoch-1.safetensors')
        assert all(np.array_equal(initial[k], zero[k]) for k in initial)
        assert all(np.array_equal(initial[k], trained[k]) for k in initial if k not in inventory[lang])
        for component in ('heads.', 'base.'):
            assert any(not np.array_equal(initial[k], trained[k]) for k in inventory[lang] if k.startswith(component))
        assert np.array_equal(np.load(output/(lang+'-initial-new-logits.npy')), data[lang]['new_initial_logits'])
        occurrences = [json.loads(line) for line in (output/(lang+'-occurrences.jsonl')).read_text().splitlines()]
        assert [r['case_id'] for r in occurrences] == [r['case_id'] for r in data[lang][arm]]
