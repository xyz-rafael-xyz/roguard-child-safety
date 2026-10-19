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
    entries = read_rows(cache/'inputs.jsonl.gz')
    rows = [r['row'] for r in entries if r['partition'] == 'train']
    heads = {}
    for language, family, epoch in (('ro', 'gemma4-structured-143', 10), ('uk', 'gemma4-structured-expanded-145', 10)):
        report = json.loads((root/family/'actual/report.json').read_text())
        path = root/family/'actual'/f'epoch-{epoch}.safetensors'
        expected = next(r['checkpoint_sha256'] for r in report['epochs'] if r['epoch'] == epoch)
        if not report.get('complete') or digest(path) != expected:
            raise ValueError('Initial head checkpoint differs')
        heads[language] = dict(path=str(path), sha256=expected, epoch=epoch, family=family)
    rng = random.Random(1165)
    batch_stats = []
    for epoch in (1, 2, 3):
        batches = plan_batches(rows, rng, 8)
        indices = [i for batch, _, _ in batches for i in batch]
        if sorted(indices) != list(range(len(rows))):
            raise ValueError('Minibatches do not cover every training case exactly once')
        batch_stats.append(dict(epoch=epoch, batches=len(batches), cases=len(indices),
            paired_comparisons=sum(len(v) for _, v, _ in batches),
            invariant_comparisons=sum(len(v) for _, _, v in batches), maximum_cases=max(len(v) for v, _, _ in batches)))
    study = root/'shared-tail-fit-165'
    study.mkdir(exist_ok=False)
    protocol = dict(study=165, created_utc=datetime.now(timezone.utc).isoformat(), seed=1165, epochs=3,
        boundary=24, batch_size=8, learning_rate=1e-4, optimizer='AdamW, weight_decay0.01, gradient_norm_clip1',
        lora=dict(rank=8, scale=16., dropout=0., keys=['self_attn.q_proj', 'self_attn.o_proj']),
        heads=heads, head_initialization='Separate RO143epoch10 and UK145epoch10 semantic/binary heads',
        objective='Language/label balanced binaryCE + mean six-fieldCE +0.2 paired softplus margin +0.1 mean symmetric KL across seven output distributions for identical-current-message history contrasts',
        coupling='Train-only language/label-balanced semantic pair potentials',
        selection='Initial and every half epoch; independent per-language checkpoint, output mode and threshold using unchanged99/5 development gates, including161 transitions',
        validation_modes=['binary', 'policy', 'coupled_policy'],
        cache='Verified164 shared-KV corpus; complete original prompt preserved; frozen earlier layers and per-layer inputs',
        expected_training_rows=len(rows), expected_development_rows=sum(r['partition'] == 'dev' for r in entries),
        batches=batch_stats, corpus_source_freeze_sha256=digest(cache/'source-freeze.json'),
        calibration_inferred=False, fresh_test_inferred=False, synthetic_only=True,
        independently_annotated=False, stable_release_ready=False, production_runtime_supported=False,
        publication='Requires separate frozen prior qualification, installed execution parity and reserved evaluation')
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    paths = [study/'protocol.json', cache/'source-freeze.json', cache/'inputs.jsonl.gz',
        cache/'base-manifest.json', cache/'software.json', cache/'protocol.json']
    paths += [Path(spec['path']) for spec in heads.values()]
    paths += [Path('training')/name for name in ('prepare_shared_tail_training_v9.py', 'train_shared_tail_v9.py',
        'shared_tail_training_v9.py', 'gemma4_shared_tail_v9.py', 'shared_tail_cache_v9.py',
        'budgeted_native_frontier_v9.py', 'entailment_frontier_v9.py', 'native_fields_v9.py',
        'prepare_instruction_sft_v8.py', 'extract_gemma4_qualified_scope_v9.py', 'diagnose_contextual_complementarity_v9.py')]
    paths += [Path('src/roguard/evidence.py')]
    (study/'source-freeze.json').write_text(json.dumps({str(p): digest(p) for p in paths}, indent=2)+'\n')
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    main()
