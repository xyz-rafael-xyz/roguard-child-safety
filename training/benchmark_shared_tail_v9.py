"""Pinned-weight training-only parity and gradient preflight for shared-KV replay."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import distribution, version
import json
from pathlib import Path
import time

from training.extract_gemma4_qualified_scope_v9 import digest
from training.diagnose_contextual_complementarity_v9 import read_rows
from training.gemma4_prompt_v9 import classifier_tokens
from training.gemma4_shared_tail_v9 import capture, replay
from training.gemma4_structured_v9 import portable_logits
from training.prepare_instruction_sft_v8 import target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    study = root/'shared-tail-preflight-163'
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen source or input differs: ' + path)
    base = json.loads((study/'base-manifest.json').read_text())
    if args.base.name != base['revision']:
        raise ValueError('Wrong model revision')
    for name, entry in base['files'].items():
        if digest(args.base/name) != entry['sha256']:
            raise ValueError('Base differs')
    software = json.loads((study/'software.json').read_text())
    for name, expected in software['versions'].items():
        if version(name) != expected:
            raise ValueError('Dependency differs: ' + name)
    package = Path(distribution('mlx-lm').locate_file('mlx_lm'))
    for name, expected in software['implementation_sha256'].items():
        if digest(package/name) != expected:
            raise ValueError('MLX implementation differs')
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from safetensors.numpy import load_file
    protocol = json.loads((study/'protocol.json').read_text())
    all_rows = read_rows(root/'gemma4-readout-141/inputs.jsonl.gz')
    rows = [all_rows[i] for i in protocol['reference_indices']]
    if any(r['partition'] != 'train' for r in rows):
        raise ValueError('Only training inputs permitted in this preflight')
    output = study/'actual'
    output.mkdir(exist_ok=False)
    mx.set_memory_limit(10*1024**3)
    mx.set_cache_limit(256*1024**2)
    mx.random.seed(1163)
    model, tokenizer = load(str(args.base), tokenizer_config={'trust_remote_code': False})
    model.set_dtype(mx.float32)
    labels = [tokenizer.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    if any(len(v) != 1 for v in labels):
        raise ValueError('Unexpected class labels')
    tokens = [classifier_tokens(tokenizer, r['row'], [v[0] for v in labels]) for r in rows]
    import hashlib
    if any(hashlib.sha256(json.dumps(v, separators=(',', ':')).encode()).hexdigest() != r['token_sha256'] for r, v in zip(rows, tokens)):
        raise ValueError('Prompt differs from frozen training input')
    head = load_file(root/'gemma4-structured-expanded-145/actual/epoch-10.safetensors')
    records, states = [], []
    def arrays(state):
        return [state['hidden']]+[v for pair in state['shared'].values() for v in pair]+[v for v in state['per_layer']+state['masks'] if v is not None]
    for row, ids in zip(rows, tokens):
        inputs = mx.array([ids])
        started = time.monotonic()
        full = model.language_model.model(inputs)[:, -1:, :]
        mx.eval(full)
        full_seconds = time.monotonic()-started
        record = dict(case_id=row['row']['case_id'], language=row['row']['language'], tokens=len(ids), full_seconds=full_seconds, tails={})
        for start in protocol['boundaries']:
            tick = time.monotonic()
            state = capture(model, inputs, start=start)
            mx.eval(*arrays(state))
            capture_seconds = time.monotonic()-tick
            timings = []
            for _ in range(3):
                tick = time.monotonic()
                actual = replay(model, state)
                mx.eval(actual)
                timings.append(time.monotonic()-tick)
            reference = np.asarray(full[:, 0, :])
            result = np.asarray(actual[:, 0, :])
            if not np.isfinite(reference).all() or not np.isfinite(result).all():
                raise ValueError('Nonfinite contextual representation')
            relative = float(np.max(np.abs(reference-result))/max(1., np.max(np.abs(reference))))
            logit_error = float(np.max(np.abs(portable_logits(reference, head)-portable_logits(result, head))))
            if relative > protocol['relative_feature_tolerance'] or logit_error > protocol['readout_logit_tolerance']:
                raise ValueError('Actual shared-tail output differs from full sequence')
            record['tails'][str(start)] = dict(capture_seconds=capture_seconds, replay_median_seconds=float(np.median(timings)),
                relative_feature_error=relative, readout_logit_error=logit_error, cache_bytes=sum(v.nbytes for v in arrays(state)))
            if start == protocol['adaptation_boundary']:
                states.append(state)
        records.append(record)
        print('CASE', json.dumps(record), flush=True)
        mx.clear_cache()
    model.freeze()
    lm = model.language_model
    linear_to_lora_layers(lm, len(lm.layers)-protocol['adaptation_boundary'], protocol['lora'])
    parameters = dict(tree_flatten(model.trainable_parameters()))
    if not parameters or any(not key.endswith(('.lora_a', '.lora_b')) for key in parameters):
        raise ValueError('Only tail adapters may be trainable')
    tensors = {k: mx.array(v) for k, v in head.items()}
    def logits(hidden):
        values = hidden[:, 0, :]
        values = values/mx.sqrt(mx.sum(mx.square(values), axis=1, keepdims=True))*values.shape[1]**.5
        first = values@tensors['0.weight'].T+tensors['0.bias']
        return nn.gelu(first)@tensors['3.weight'].T+tensors['3.bias']
    used = protocol['gradient_reference_positions']
    head_error = 0.
    for i in used:
        hidden = replay(model, states[i])
        actual = logits(hidden)
        mx.eval(hidden, actual)
        expected = portable_logits(np.asarray(hidden[:, 0, :]), head)
        head_error = max(head_error, float(np.max(np.abs(np.asarray(actual)-expected))))
    if head_error > protocol['readout_logit_tolerance']:
        raise ValueError('Differentiable readout differs from independent export')
    def loss(m):
        values = [nn.losses.cross_entropy(logits(replay(m, states[i]))[:, -2:], mx.array([target(rows[i]['row'])])) for i in used]
        return mx.stack(values).mean()
    started = time.monotonic()
    before, gradients = nn.value_and_grad(model, loss)(model)
    mx.eval(before, gradients)
    if not bool(mx.isfinite(before).item()):
        raise ValueError('Nonfinite adapter loss')
    gradient_arrays = list(dict(tree_flatten(gradients)).values())
    if not all(bool(mx.all(mx.isfinite(v)).item()) for v in gradient_arrays):
        raise ValueError('Nonfinite adapter gradient')
    norm = float(mx.sqrt(sum(mx.sum(v*v) for v in gradient_arrays)).item())
    if not norm > 0:
        raise ValueError('No adaptation signal reached tail adapters')
    optimizer = optim.SGD(learning_rate=protocol['learning_rate'])
    optimizer.update(model, gradients)
    mx.eval(model.trainable_parameters(), optimizer.state)
    after = loss(model)
    mx.eval(after)
    if not bool(mx.isfinite(after).item()):
        raise ValueError('Nonfinite updated adapter loss')
    step_seconds = time.monotonic()-started
    post_error = 0.
    for i in used:
        full = lm.model(mx.array([tokens[i]]))[:, -1:, :]
        actual = replay(model, states[i])
        mx.eval(full, actual)
        reference, result = np.asarray(full), np.asarray(actual)
        if not np.isfinite(reference).all() or not np.isfinite(result).all():
            raise ValueError('Nonfinite adapted representation')
        post_error = max(post_error, float(np.max(np.abs(reference-result))/max(1., np.max(np.abs(reference)))))
    if post_error > protocol['relative_feature_tolerance']:
        raise ValueError('Adapted full sequence and cached replay differ')
    report = dict(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), records=records,
        gradient=dict(parameters=sum(v.size for v in parameters.values()), tensors=len(parameters), norm=norm,
            differentiable_readout_logit_error=head_error,
            before_loss=float(before.item()), after_loss=float(after.item()), seconds=step_seconds,
            post_update_relative_error=post_error, loss_decreased=bool(after.item() < before.item())),
        peak_mlx_bytes=mx.get_peak_memory(), only_training_inferred=True, calibration_inferred=False,
        fresh_test_inferred=False, accuracy_experiment=False, adapter_discarded=True,
        synthetic_only=True, stable_release_ready=False, source_freeze_sha256=digest(study/'source-freeze.json'))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('FINISHED', json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
