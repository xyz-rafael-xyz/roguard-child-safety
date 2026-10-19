"""Freeze adapter training before the shared-tail corpus is evaluated."""
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.shared_tail_training_v9 import plan_batches


def main():
    root = Path('research/v9-stabilization')
    cache = root/'shared-tail-corpus-164'
    for path, expected in json.loads((cache/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen corpus source differs: ' + path)
    caches = [root/name for name in ('shared-tail-corpus-164', 'grounding-tail-corpus-174', 'referent-tail-corpus-176')]
    entries = []
    for corpus in caches:
        for path, expected in json.loads((corpus/'source-freeze.json').read_text()).items():
            if digest(path) != expected:
                raise ValueError('Frozen corpus changed: '+path)
        entries.extend(read_rows(corpus/'inputs.jsonl.gz'))
    if len({r['token_sha256'] for r in entries}) != len(entries):
        raise ValueError('Duplicate inputs across old and new populations')
    previous = root/'shared-tail-fit-165'
    old = json.loads((previous/'protocol.json').read_text())
    from training.shared_tail_qualification_v9 import candidate_bindings
    specs, initial_bindings = candidate_bindings(previous)
    heads = old['heads']
    initial = {language:dict(path=str(previous/'actual'/f'{language}-selected.safetensors'),
        sha256=specs[language]['checkpoint_sha256']) for language in ('ro', 'uk')}
    batch_stats = {}
    for language in ('ro', 'uk'):
        rows = [r['row'] for r in entries if r['partition'] == 'train' and r['row']['language'] == language]
        rng = random.Random(1177+(0 if language == 'ro' else 1))
        batch_stats[language] = []
        for epoch in (1, 2):
            batches = plan_batches(rows, rng, 8)
            indices = [i for batch, _, _ in batches for i in batch]
            if sorted(indices) != list(range(len(rows))):
                raise ValueError('Minibatches must cover each training input once')
            batch_stats[language].append(dict(epoch=epoch, cases=len(indices), batches=len(batches),
                paired_comparisons=sum(len(v) for _, v, _ in batches),
                invariant_comparisons=sum(len(v) for _, _, v in batches)))
    study = root/'grounding-tail-fit-177'
    study.mkdir(exist_ok=False)
    protocol = dict(study=177, created_utc=datetime.now(timezone.utc).isoformat(), seed=1177, epochs=2,
        boundary=24, batch_size=8, learning_rate=5e-5, optimizer='AdamW, weight_decay0.01, gradient_norm_clip1',
        lora=dict(rank=8, scale=16., dropout=0., keys=['self_attn.q_proj', 'self_attn.o_proj']),
        heads=heads, initial=initial, caches=[str(c) for c in caches], head_initialization='Full selected165 language checkpoint; independent continuation per language',
        objective='Equal total old/new training mass; each half label-balanced binaryCE + mean six-fieldCE +0.2 paired softplus margin +0.1 mean symmetric KL across seven output distributions for identical-current-message history contrasts',
        coupling='Preserve165 training-fitted semantic pair potentials',
        selection='Initial and every half epoch; minimize maximum normalized error-budget utilization (100*FN/P and20*FP/N), then sum utilization, then errors, over all DEV groups; independent language/mode/threshold; unchanged99/5 gates',
        validation_modes=['binary', 'policy', 'coupled_policy'],
        cache='Composite verified164/174/176 caches; preserve original per-corpus DEV batch8 boundaries',
        expected_training_rows=sum(r['partition'] == 'train' for r in entries), expected_development_rows=sum(r['partition'] == 'dev' for r in entries),
        batches=batch_stats, corpus_source_freeze_sha256=digest(cache/'source-freeze.json'),
        calibration_inferred=False, fresh_test_inferred=False, synthetic_only=True,
        independently_annotated=False, stable_release_ready=False, production_runtime_supported=False,
        publication='Requires separate frozen prior qualification, installed execution parity and reserved evaluation')
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    paths = [study/'protocol.json', cache/'source-freeze.json', cache/'inputs.jsonl.gz',
        cache/'base-manifest.json', cache/'software.json', cache/'protocol.json']
    paths += [Path(spec['path']) for spec in heads.values()]
    paths += [Path('training')/name for name in ('prepare_grounding_tail_training_v9.py', 'train_grounding_tail_v9.py', 'composite_tail_cache_v9.py', 'balanced_headroom_frontier_v9.py', 'train_shared_tail_v9.py', 'shared_tail_qualification_v9.py',
        'shared_tail_training_v9.py', 'gemma4_shared_tail_v9.py', 'shared_tail_cache_v9.py',
        'budgeted_native_frontier_v9.py', 'entailment_frontier_v9.py', 'native_fields_v9.py',
        'prepare_instruction_sft_v8.py', 'extract_gemma4_qualified_scope_v9.py', 'diagnose_contextual_complementarity_v9.py')]
    paths += [Path('src/roguard/evidence.py')]
    paths += [corpus/name for corpus in caches for name in ('source-freeze.json', 'inputs.jsonl.gz', 'protocol.json')]
    paths += [Path(path) for path in initial_bindings]
    (study/'source-freeze.json').write_text(json.dumps({str(p): digest(p) for p in paths}, indent=2)+'\n')
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    main()
