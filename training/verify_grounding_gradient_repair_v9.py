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
from training.shared_tail_training_v9 import plan_batches, supervised_loss
from training.serial_tail_training_v9 import make_classifier


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
    output = root/'grounding-gradient-repair-185'
    output.mkdir(exist_ok=False)
    (output/'source-freeze.json').write_text(json.dumps({str(Path(__file__)):digest(__file__),str(study/'source-freeze.json'):digest(study/'source-freeze.json'),'training/serial_tail_training_v9.py':digest('training/serial_tail_training_v9.py')},indent=2)+'\n')
    model.train()
    def loss_fn(m, states, languages, y, factors, w, pairs, invariant):
        return supervised_loss(m(states, languages), y, factors, w, pairs, invariant)
    value_and_grad = nn.value_and_grad(model, loss_fn)
    batches = plan_batches(rows, rng, protocol['batch_size'])
    result=[]
    for batch_number in (0,1,2,3):
        indices,pairs,invariant=batches[batch_number]
        states=[to_mlx(s) for s in reader.read_many([training_indices[i] for i in indices])]
        arguments=(states,[language]*len(indices),mx.array(review[indices]),mx.array(fields[indices]),mx.array(weights[indices]),pairs,invariant)
        records=[]
        for repeat in range(3):
            loss,grads=value_and_grad(model,*arguments)
            mx.eval(loss,grads)
            gradients={name:np.asarray(value).copy() for name,value in tree_flatten(grads)}
            name=max(gradients,key=lambda n:float(np.max(np.abs(gradients[n]))))
            if batch_number == 1: name='base.model.layers.24.self_attn.o_proj.lora_b'
            g=gradients[name];coordinate=(2,1038) if batch_number == 1 else np.unravel_index(np.argmax(np.abs(g)),g.shape)
            records.append(dict(repeat=repeat,loss=float(loss.item()),norm=float(np.sqrt(sum(np.square(g.astype(np.float64)).sum() for g in gradients.values()))),largest=name,coordinate=list(map(int,coordinate)),derivative=float(g[coordinate])))
        values=dict(tree_flatten(model.trainable_parameters()))
        original=values[name]
        index=tuple(map(int,coordinate))
        numerical=[]
        for epsilon in (1e-3,1e-4,1e-5,1e-6):
            scores=[]
            for sign in (1,-1):
                changed=original.at[index].add(sign*epsilon)
                model.load_weights([(name,changed)],strict=False)
                value=loss_fn(model,*arguments)
                mx.eval(value)
                scores.append(float(value.item()))
            numerical.append(dict(epsilon=epsilon,plus=scores[0],minus=scores[1],derivative=(scores[0]-scores[1])/(2*epsilon)))
        model.load_weights([(name,original)],strict=False)
        result.append(dict(batch=batch_number,repeats=records,numerical=numerical))
        print('DIFFERENCE',json.dumps(result[-1]),flush=True)
        (output/'report.json').write_text(json.dumps(dict(complete=False,batches=result),indent=2)+'\n')
        mx.clear_cache()
    (output/'report.json').write_text(json.dumps(dict(complete=True,batches=result,training_only=True,fresh_test_inferred=False),indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    run_language('ro',parser.parse_args())
