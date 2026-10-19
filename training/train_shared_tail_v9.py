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

from training.budgeted_native_frontier_v9 import frontier
from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.native_fields_v9 import SIZES, fit_couplings, policy_mass, targets as field_targets
from training.prepare_instruction_sft_v8 import target
from training.shared_tail_cache_v9 import read_state, to_mlx
from training.shared_tail_training_v9 import make_classifier, plan_batches, supervised_loss


class Reader:
    def __init__(self, study, entries):
        self.study, self.entries = study, entries
        self.freeze = digest(study/'source-freeze.json')
        report = json.loads((study/'actual/report.json').read_text())
        if not report.get('complete') or report['rows'] != len(entries) or report['source_freeze_sha256'] != self.freeze:
            raise ValueError('Cache extraction must be complete and bound to these inputs')
        self.records, self.verified = {}, {}
        offset = 0
        for chunk in report['chunks']:
            path = study/'actual'/f"chunk-{chunk['start']:06d}.json"
            if digest(path) != chunk['sha256'] or chunk['start'] != offset:
                raise ValueError('Cache manifest differs or has a gap')
            record = json.loads(path.read_text())
            if record['source_freeze_sha256'] != self.freeze or record['stop'] != chunk['stop']:
                raise ValueError('Cache manifest source differs')
            for item in record['states']:
                index = item['index']
                if index != offset or item['token_sha256'] != entries[index]['token_sha256']:
                    raise ValueError('Cached input ordering differs')
                self.records[index] = item
                offset += 1
            if offset != chunk['stop']:
                raise ValueError('Incomplete cached chunk')
        if offset != len(entries):
            raise ValueError('Incomplete cache corpus')

    def read_many(self, indices):
        states = []
        for index in indices:
            path = self.study/'actual/states'/f'state-{index:06d}.npz'
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
            if self.verified.get(index) != signature:
                if stat.st_size != self.records[index]['bytes'] or digest(path) != self.records[index]['sha256']:
                    raise ValueError('Cached tensor content differs')
                self.verified[index] = signature
            states.append(read_state(path, expected_identity=dict(token_sha256=self.entries[index]['token_sha256'], source_freeze_sha256=self.freeze)))
        return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    study = root/'shared-tail-fit-165'
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
    rng = random.Random(protocol['seed'])
    entries = read_rows(cache/'inputs.jsonl.gz')
    reader = Reader(cache, entries)
    training_indices = [i for i, r in enumerate(entries) if r['partition'] == 'train']
    rows = [entries[i]['row'] for i in training_indices]
    groups = [(r['language'], target(r)) for r in rows]
    frequencies = Counter(groups)
    if len(frequencies) != 4:
        raise ValueError('Both language/label strata required')
    weights = np.array([len(rows)/(4*frequencies[g]) for g in groups], dtype=np.float32)
    review = np.array([target(r) for r in rows], dtype=np.int32)
    fields = np.array(field_targets(rows), dtype=np.int32)
    couplings = fit_couplings(rows, weights.tolist())
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
    output = study/'actual'
    output.mkdir(exist_ok=False)
    parents = {str(cache/'actual/report.json'): digest(cache/'actual/report.json')}
    parents.update({str(cache/'actual'/f"chunk-{r['start']:06d}.json"): r['sha256'] for r in json.loads((cache/'actual/report.json').read_text())['chunks']})
    (output/'parent-freeze.json').write_text(json.dumps(parents, indent=2)+'\n')
    report = dict(complete=False, epochs=[], selected={}, languages={}, training_rows=len(rows),
        trainable_parameters=sum(v.size for v in parameters.values()), source_freeze_sha256=digest(study/'source-freeze.json'),
        parent_freeze_sha256=digest(output/'parent-freeze.json'), calibration_inferred=False,
        fresh_test_inferred=False, synthetic_only=True, independently_annotated=False,
        stable_release_ready=False, production_runtime_supported=False)
    def save_report():
        (output/'development-progress.json').write_text(json.dumps(report, indent=2)+'\n')
    def evaluate(epoch):
        model.eval()
        checkpoint = output/f'epoch-{epoch}.safetensors'
        mx.save_safetensors(str(checkpoint), dict(tree_flatten(model.trainable_parameters())))
        checkpoint_hash = digest(checkpoint)
        epoch_result = dict(epoch=epoch, checkpoint_sha256=checkpoint_hash, languages={})
        for language in ('ro', 'uk'):
            indices = [i for i, r in enumerate(entries) if r['partition'] == 'dev' and r['row']['language'] == language]
            values = []
            for start in range(0, len(indices), protocol['batch_size']):
                chosen = indices[start:start+protocol['batch_size']]
                states = [to_mlx(s) for s in reader.read_many(chosen)]
                logits = model(states, [language]*len(chosen))
                mx.eval(logits)
                values.append(np.asarray(logits).copy())
                del states, logits
                mx.clear_cache()
            logits = np.concatenate(values).astype(np.float64)
            if not np.isfinite(logits).all():
                raise ValueError('Nonfinite development logits')
            def softmax(v):
                p = np.exp(v-v.max(axis=1, keepdims=True))
                return p/p.sum(axis=1, keepdims=True)
            factors = [softmax(v) for v in np.split(logits[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)]
            variants = dict(binary=softmax(logits[:, -2:])[:, 1], policy=policy_mass(factors), coupled_policy=policy_mass(factors, couplings))
            comparisons = {}
            for mode, scores in variants.items():
                population = [dict(case_id=entries[i]['row']['case_id'], language=language, group=entries[i]['group'],
                    gold=target(entries[i]['row']), score=float(value)) for i, value in zip(indices, scores)]
                result = frontier(population)
                comparisons[mode] = result
                old = report['selected'].get(language)
                if old is None or tuple(result['rank']) < tuple(old['frontier']['rank']):
                    selection = dict(epoch=epoch, mode=mode, frontier=result, checkpoint_sha256=checkpoint_hash)
                    report['selected'][language] = selection
                    shutil.copyfile(checkpoint, output/f'{language}-selected.safetensors')
                    artifact = dict(format='roguard-shared-tail-adapter-experiment-v1', language=language,
                        base_revision=manifest['revision'], base_manifest_sha256=digest(cache/'base-manifest.json'),
                        source_freeze_sha256=digest(study/'source-freeze.json'), boundary=protocol['boundary'],
                        lora=protocol['lora'], checkpoint_sha256=checkpoint_hash, mode=mode,
                        couplings=couplings, threshold=result['threshold'], score_is_calibrated=False)
                    (output/f'{language}-review.json').write_text(json.dumps(artifact, indent=2)+'\n')
                    for row in population:
                        row['predicted'] = int(row['score'] >= result['threshold'])
                    (output/f'{language}-dev-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in population))
            epoch_result['languages'][language] = comparisons
            print('DEVELOPMENT', epoch, language, json.dumps(report['selected'][language]), flush=True)
        report['epochs'].append(epoch_result)
        save_report()
        model.train()
    evaluate(0)
    optimizer = optim.AdamW(learning_rate=protocol['learning_rate'], weight_decay=.01)
    batches_by_epoch = [plan_batches(rows, rng, protocol['batch_size']) for _ in range(protocol['epochs'])]
    total_steps = sum(map(len, batches_by_epoch))
    step = 0
    def loss_fn(m, states, languages, y, factors, w, pairs, invariant):
        return supervised_loss(m(states, languages), y, factors, w, pairs, invariant)
    value_and_grad = nn.value_and_grad(model, loss_fn)
    with ThreadPoolExecutor(max_workers=1) as pool:
        for epoch, batches in enumerate(batches_by_epoch, 1):
            future = pool.submit(reader.read_many, [training_indices[i] for i in batches[0][0]])
            started = time.monotonic()
            losses = []
            for batch_index, (indices, pairs, invariant) in enumerate(batches):
                states = [to_mlx(s) for s in future.result()]
                if batch_index+1 < len(batches):
                    future = pool.submit(reader.read_many, [training_indices[i] for i in batches[batch_index+1][0]])
                warm = max(1, int(.05*total_steps))
                scale = min((step+1)/warm, max(.1, (total_steps-step)/(total_steps-warm)))
                optimizer.learning_rate = protocol['learning_rate']*scale
                loss, gradients = value_and_grad(model, states, [rows[i]['language'] for i in indices],
                    mx.array(review[indices]), mx.array(fields[indices]), mx.array(weights[indices]), pairs, invariant)
                gradients, norm = optim.clip_grad_norm(gradients, 1.)
                mx.eval(loss, norm, gradients)
                if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):
                    raise ValueError('Nonfinite training loss or gradient norm')
                optimizer.update(model, gradients)
                mx.eval(model.trainable_parameters(), optimizer.state)
                losses.append(float(loss.item()))
                step += 1
                del states, gradients
                mx.clear_cache()
                if batch_index % 100 == 0:
                    print('TRAIN', epoch, batch_index, '/', len(batches), 'loss', losses[-1], 'seconds', round(time.monotonic()-started, 2), flush=True)
                if batch_index+1 in (len(batches)//2, len(batches)):
                    evaluate(epoch-.5 if batch_index+1 < len(batches) else epoch)
            report.setdefault('training', []).append(dict(epoch=epoch, steps=len(batches), mean_loss=float(np.mean(losses)), seconds=time.monotonic()-started))
            save_report()
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
        development_passes=all(v['frontier']['counts_pass'] for v in report['selected'].values()),
        peak_mlx_bytes=mx.get_peak_memory())
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('FINISHED', json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
