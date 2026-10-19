"""Select a Ukrainian checkpoint by DEV error budgets and score separation.

Romanian is carried unchanged. No prior/calibration or reserved scores are read.
The complete candidate list and selection rule are frozen before inference.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import distribution, version
import json
from pathlib import Path
import shutil

from training.composite_tail_cache_v9 import CompositeReader
from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.prepare_instruction_sft_v8 import target
from training.separation_frontier_v9 import frontier
from training.shared_tail_cache_v9 import to_mlx
from training.shared_tail_qualification_v9 import candidate_bindings, compare_replay, scores_from_logits
from training.shared_tail_training_v9 import make_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    previous, study = args.candidate, args.study
    specs, bindings = candidate_bindings(previous)
    old = json.loads((previous/'actual/report.json').read_text())
    protocol = json.loads((previous/'protocol.json').read_text())
    checkpoints = [dict(epoch=e['epoch'], path=str(previous/'actual/uk'/f"epoch-{e['epoch']}.safetensors"),
        sha256=e['checkpoint_sha256']) for e in old['languages']['uk']['epochs']]
    for checkpoint in checkpoints:
        if digest(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Candidate checkpoint changed')
    protocol.update(study=int(study.name.rsplit('-', 1)[1]), created_utc=datetime.now(timezone.utc).isoformat(),
        selection='Retain Romanian186 unchanged; evaluate every Ukrainian186 checkpoint and all three original modes on DEV only; minimize original error-budget rank then maximize zero-error additive score separation; exact remaining ties retain earlier checkpoint/mode',
        selection_candidates=checkpoints, training_performed=False,
        prior_results_used_for_selection=False,
        motivation='Study187 exposed poor generalization despite zero DEV errors; earlier selector retained the first perfect checkpoint without considering separation',
        margin_limit='Development score separation is not a calibrated uncertainty bound or a generalization guarantee')
    study.mkdir(exist_ok=False)
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    bindings.update({str(study/'protocol.json'): digest(study/'protocol.json')})
    bindings.update({c['path']: c['sha256'] for c in checkpoints})
    paths = [Path('training')/name for name in ('select_separated_tail_v9.py', 'separation_frontier_v9.py',
        'balanced_headroom_frontier_v9.py', 'composite_tail_cache_v9.py', 'shared_tail_qualification_v9.py',
        'shared_tail_training_v9.py', 'shared_tail_cache_v9.py', 'train_shared_tail_v9.py', 'native_fields_v9.py',
        'entailment_frontier_v9.py', 'prepare_instruction_sft_v8.py')]
    bindings.update({str(p): digest(p) for p in paths})
    (study/'source-freeze.json').write_text(json.dumps(bindings, indent=2)+'\n')
    freeze = digest(study/'source-freeze.json')
    output = study/'actual'
    output.mkdir()
    cache = Path(protocol['caches'][0])
    manifest = json.loads((cache/'base-manifest.json').read_text())
    if args.base.name != manifest['revision']:
        raise ValueError('Base revision differs')
    for name, record in manifest['files'].items():
        if digest(args.base/name) != record['sha256']:
            raise ValueError('Base file changed')
    software = json.loads((cache/'software.json').read_text())
    for name, expected in software['versions'].items():
        if version(name) != expected:
            raise ValueError('Dependency changed')
    package = Path(distribution('mlx-lm').locate_file('mlx_lm'))
    for name, expected in software['implementation_sha256'].items():
        if digest(package/name) != expected:
            raise ValueError('MLX implementation changed')
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from safetensors.numpy import load_file
    mx.set_memory_limit(10*1024**3)
    mx.set_cache_limit(256*1024**2)
    reader = CompositeReader([Path(p) for p in protocol['caches']])
    (output/'parent-freeze.json').write_text(json.dumps(reader.parents, indent=2)+'\n')
    base, _ = load(str(args.base), tokenizer_config={'trust_remote_code': False})
    base.set_dtype(mx.float32)
    model = make_classifier(base.language_model,
        {lang: load_file(Path(head['path'])) for lang, head in protocol['heads'].items()},
        boundary=protocol['boundary'], lora=protocol['lora'])
    del base
    model.eval()
    parameters = dict(tree_flatten(model.trainable_parameters()))
    batches = list(reader.development_batches('uk', protocol['batch_size']))
    indices = [i for batch in batches for i in batch]
    report = dict(complete=False, selected={'ro': old['selected']['ro']}, comparisons=[],
        source_freeze_sha256=freeze, parent_freeze_sha256=digest(output/'parent-freeze.json'),
        training_performed=False, prior_results_used_for_selection=False, calibration_inferred=False,
        fresh_test_inferred=False, synthetic_only=True, independently_annotated=False,
        stable_release_ready=False, production_runtime_supported=False)
    for suffix in ('selected.safetensors', 'dev-predictions.jsonl'):
        shutil.copyfile(previous/'actual'/('ro-'+suffix), output/('ro-'+suffix))
    artifact = dict(specs['ro'], source_freeze_sha256=freeze)
    (output/'ro-review.json').write_text(json.dumps(artifact, indent=2)+'\n')
    for checkpoint in checkpoints:
        weights = mx.load(checkpoint['path'])
        if set(weights) != set(parameters) or any(v.shape != parameters[k].shape or v.dtype != mx.float32
                or not bool(mx.all(mx.isfinite(v)).item()) for k, v in weights.items()):
            raise ValueError('Checkpoint inventory differs')
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.trainable_parameters())
        del weights
        values = []
        for chosen in batches:
            states = [to_mlx(s) for s in reader.read_many(chosen)]
            logits = model(states, ['uk']*len(chosen))
            mx.eval(logits)
            values.append(np.asarray(logits).copy())
            del states, logits
            mx.clear_cache()
        logits = np.concatenate(values).astype(np.float64)
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite DEV logits')
        comparison = dict(epoch=checkpoint['epoch'], checkpoint_sha256=checkpoint['sha256'], modes={})
        for mode in protocol['validation_modes']:
            scores = scores_from_logits(logits, dict(specs['uk'], mode=mode))
            population = [dict(case_id=reader.entries[i]['row']['case_id'], language='uk',
                group=reader.entries[i]['group'], gold=target(reader.entries[i]['row']), score=float(value))
                for i, value in zip(indices, scores)]
            if checkpoint['sha256'] == specs['uk']['checkpoint_sha256'] and mode == specs['uk']['mode']:
                report['original_selection_replay'] = compare_replay(population,
                    read_rows(previous/'actual/uk-dev-predictions.jsonl'), specs['uk']['threshold'], 1e-5)
            result = frontier(population)
            comparison['modes'][mode] = result
            chosen = report['selected'].get('uk')
            if chosen is None or tuple(result['rank']) < tuple(chosen['frontier']['rank']):
                report['selected']['uk'] = dict(epoch=checkpoint['epoch'], mode=mode,
                    checkpoint_sha256=checkpoint['sha256'], frontier=result)
                shutil.copyfile(checkpoint['path'], output/'uk-selected.safetensors')
                artifact = dict(specs['uk'], source_freeze_sha256=freeze, mode=mode,
                    threshold=result['threshold'], checkpoint_sha256=checkpoint['sha256'])
                (output/'uk-review.json').write_text(json.dumps(artifact, indent=2)+'\n')
                for row in population:
                    row['predicted'] = int(row['score'] >= result['threshold'])
                (output/'uk-dev-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in population))
        report['comparisons'].append(comparison)
        (output/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        print('DEV_SELECTION', json.dumps(dict(epoch=checkpoint['epoch'], selected=report['selected']['uk'])), flush=True)
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError('Selection source changed')
    report.update(complete=True, development_passes=all(s['frontier']['counts_pass'] for s in report['selected'].values()),
        finished_utc=datetime.now(timezone.utc).isoformat(), peak_mlx_bytes=mx.get_peak_memory())
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    candidate_bindings(study)
    print('COMPLETE', json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
