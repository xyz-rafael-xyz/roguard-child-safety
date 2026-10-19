"""Bilingual semantic adaptation through a frozen-body, shared-KV Gemma4 tail."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import distribution, version
import json
from pathlib import Path
import random
import re
import shutil
import time

from training.balanced_headroom_frontier_v9 import frontier
from training.composite_tail_cache_v9 import CompositeReader
from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.native_fields_v9 import SIZES, fit_couplings, policy_mass, targets as field_targets
from training.prepare_instruction_sft_v8 import target
from training.shared_tail_cache_v9 import read_state, to_mlx
from training.shared_tail_training_v9 import make_classifier, plan_batches, supervised_loss


def run_language(language, args):
    root = Path('research/v9-stabilization')
    study = root/'grounding-tail-fit-177'
    cache = root/'shared-tail-corpus-164'
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen source or input differs: ' + path)
    protocol = json.loads((study/'protocol.json').read_text())
    manifest = json.loads((cache/'base-manifest.json').read_text())
    if args.base.name != manifest['revision']:
        raise ValueError('Wrong base revision')
    for name, record in manifest['files'].items():
        if digest(args.base/name) != record['sha256']:
            raise ValueError('Pinned base differs')
    software = json.loads((cache/'software.json').read_text())
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
    from safetensors.numpy import load_file
    mx.set_memory_limit(10*1024**3)
    mx.set_cache_limit(256*1024**2)
    mx.random.seed(protocol['seed'])
    rng = random.Random(protocol['seed'] + (0 if language == 'ro' else 1))
    reader = CompositeReader([Path(p) for p in protocol['caches']])
    entries = reader.entries
    training_indices = [i for i, r in enumerate(entries) if r['partition'] == 'train' and r['row']['language'] == language]
    rows = [entries[i]['row'] for i in training_indices]
    groups = [('old' if reader.locations[i][0] == 0 else 'new', target(r)) for i, r in zip(training_indices, rows)]
    frequencies = Counter(groups)
    if len(frequencies) != 4:
        raise ValueError('Old/new by label strata required')
    weights = np.array([len(rows)/(4*frequencies[g]) for g in groups], dtype=np.float32)
    review = np.array([target(r) for r in rows], dtype=np.int32)
    fields = np.array(field_targets(rows), dtype=np.int32)
    couplings = json.loads((root/'shared-tail-fit-165/actual'/f'{language}-review.json').read_text())['couplings']
    base, _ = load(str(args.base), tokenizer_config={'trust_remote_code': False})
    base.set_dtype(mx.float32)
    tensors = {lang: load_file(Path(spec['path'])) for lang, spec in protocol['heads'].items()}
    model = make_classifier(base.language_model, tensors, boundary=protocol['boundary'], lora=protocol['lora'])
    del base
    parameters = dict(tree_flatten(model.trainable_parameters()))
    for name in parameters:
        if name.startswith('heads.ro.') or name.startswith('heads.uk.'):
            continue
        match = re.fullmatch(r'base\.model\.layers\.(\d+)\.self_attn\.(q_proj|o_proj)\.lora_[ab]', name)
        if not match or int(match.group(1)) < protocol['boundary']:
            raise ValueError('Frozen-body parameter became trainable: ' + name)
    initial = protocol['initial'][language]
    weights_initial = mx.load(initial['path'])
    if set(weights_initial) != set(parameters):
        raise ValueError('Initial adapter inventory differs')
    for name, value in weights_initial.items():
        if value.shape != parameters[name].shape or value.dtype != mx.float32 or not bool(mx.all(mx.isfinite(value)).item()):
            raise ValueError('Invalid initial adapter tensor: '+name)
    model.load_weights(list(weights_initial.items()), strict=False)
    mx.eval(model.trainable_parameters())
    del weights_initial
    output = root/'grounding-gradient-diagnostic-182'
    output.mkdir(exist_ok=False)
    freeze = {str(Path(__file__)): digest(__file__), str(study/'source-freeze.json'): digest(study/'source-freeze.json')}
    (output/'source-freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    model.train()
    optimizer = optim.AdamW(learning_rate=protocol['learning_rate'], weight_decay=.01)
    batches_by_epoch = [plan_batches(rows, rng, protocol['batch_size']) for _ in range(protocol['epochs'])]
    total_steps = sum(map(len, batches_by_epoch))
    def loss_fn(m, states, languages, y, factors, w, pairs, invariant):
        return supervised_loss(m(states, languages), y, factors, w, pairs, invariant)
    value_and_grad = nn.value_and_grad(model, loss_fn)
    report = dict(steps=[], training_only=True, calibration_inferred=False, fresh_test_inferred=False)
    for step, (indices, pairs, invariant) in enumerate(batches_by_epoch[0][:32]):
        states = [to_mlx(s) for s in reader.read_many([training_indices[i] for i in indices])]
        warm = max(1, int(.05*total_steps))
        scale = min((step+1)/warm, max(.1, (total_steps-step)/(total_steps-warm)))
        optimizer.learning_rate = protocol['learning_rate']*scale
        logits = model(states, [language]*len(indices))
        loss, gradients = value_and_grad(model, states, [language]*len(indices),
            mx.array(review[indices]), mx.array(fields[indices]), mx.array(weights[indices]), pairs, invariant)
        mx.eval(logits, loss, gradients)
        raw = {name: np.asarray(value).copy() for name, value in tree_flatten(gradients)}
        bad = [name for name, value in raw.items() if not np.isfinite(value).all()]
        norm64 = float(np.sqrt(sum(np.square(value.astype(np.float64)).sum() for value in raw.values())))
        clipped, norm = optim.clip_grad_norm(gradients, 1.)
        mx.eval(norm, clipped)
        record = dict(step=step, loss=float(loss.item()), norm=float(norm.item()), norm64=norm64,
            bad_gradients=bad, logits_finite=bool(mx.all(mx.isfinite(logits)).item()),
            logits_range=[float(mx.min(logits).item()),float(mx.max(logits).item())],
            cases=[rows[i]['case_id'] for i in indices],pairs=pairs,invariant=invariant)
        report['steps'].append(record)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('STEP',json.dumps(record),flush=True)
        if bad or not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):
            break
        optimizer.update(model, clipped)
        mx.eval(model.trainable_parameters(), optimizer.state)
        del states, gradients, clipped, logits
        mx.clear_cache()
    report['complete'] = True
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    run_language('ro',parser.parse_args())
