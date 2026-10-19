"""Development-only selection of a contextual check after frozen agreement."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from training.budgeted_native_frontier_v9 import frontier
from training.diagnose_contextual_complementarity_v9 import read_rows
from training.gemma4_structured_v9 import portable_logits
from training.native_fields_v9 import SIZES, fit_couplings, policy_mass
from training.prepare_instruction_sft_v8 import target
from training.score_gemma4_qualified_scope_v9 import load_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    from safetensors.numpy import load_file
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    root = Path('research/v9-stabilization')
    study = root/'conditional-contextual-155'
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen source or input differs: ' + path)
    protocol = json.loads((study/'protocol.json').read_text())
    output = study/'actual'
    output.mkdir(exist_ok=False)
    parent = root/'gemma4-readout-141'
    entries = read_rows(parent/'inputs.jsonl.gz')
    features = load_features(parent, entries)
    original = [r['row'] for r in entries if r['partition'] == 'train']
    additions = [r['row'] for r in read_rows(root/'gemma4-expansion-144/inputs.jsonl.gz')]
    coupling = {}
    hashes = {}
    for family in protocol['semantic_families']:
        rows = original if family == 'gemma4-structured-143' else original+additions
        groups = [(r['language'], target(r)) for r in rows]
        counts = Counter(groups)
        coupling[family] = fit_couplings(rows, [len(rows)/(4*counts[g]) for g in groups])
        path = root/family/'actual/report.json'
        report = json.loads(path.read_text())
        if not report.get('complete'):
            raise ValueError('Incomplete parent')
        hashes[str(path)] = digest(path)
        for epoch in protocol['epochs']:
            checkpoint = root/family/'actual'/f'epoch-{epoch}.safetensors'
            expected = next(r['checkpoint_sha256'] for r in report['epochs'] if r['epoch'] == epoch)
            if digest(checkpoint) != expected:
                raise ValueError('Semantic checkpoint differs')
            hashes[str(checkpoint)] = expected
    (output/'parent-freeze.json').write_text(json.dumps(hashes, indent=2)+'\n')
    ix = [i for i, r in enumerate(entries) if r['partition'] == 'dev' and r['row']['language'] == 'ro']
    dev = [entries[i] for i in ix]
    lookup = {r['case_id']: r for r in read_rows(root/'scope-agreement-154/actual/ro-dev-predictions.jsonl')}
    if set(lookup) != {r['row']['case_id'] for r in dev}:
        raise ValueError('Agreement population differs')
    for r in dev:
        p = lookup[r['row']['case_id']]
        if p['gold'] != target(r['row']) or p['group'].replace('scope_dev', 'scope') != r['group'].replace('scope_dev', 'scope'):
            raise ValueError('Agreement labels or groups differ')
    mask = np.array([lookup[r['row']['case_id']]['predicted'] for r in dev], dtype=np.float64)
    def population(values):
        return [dict(case_id=r['row']['case_id'], language='ro', group=r['group'],
                     gold=target(r['row']), score=float(v)) for r, v in zip(dev, values)]
    baseline = frontier(population(mask))
    best = tuple(baseline['rank'])
    selected = dict(kind='retained154', frontier=baseline)
    comparisons = []
    selected_predictions = population(mask)
    for family in protocol['semantic_families']:
        for epoch in protocol['epochs']:
            checkpoint = root/family/'actual'/f'epoch-{epoch}.safetensors'
            logits = portable_logits(features[ix], load_file(checkpoint)).astype(np.float64)
            def softmax(values):
                exp = np.exp(values-values.max(axis=1, keepdims=True))
                return exp/exp.sum(axis=1, keepdims=True)
            fields = [softmax(v) for v in np.split(logits[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)]
            variants = dict(binary=softmax(logits[:, -2:])[:, 1], policy=policy_mass(fields),
                            coupled_policy=policy_mass(fields, coupling[family]))
            for mode, values in variants.items():
                predictions = population(mask*values)
                result = frontier(predictions)
                candidate = dict(kind='conditional', semantic_family=family, epoch=epoch,
                    semantic_mode=mode, semantic_checkpoint_sha256=digest(checkpoint), frontier=result)
                comparisons.append(candidate)
                if tuple(result['rank']) < best:
                    if result['threshold'] <= 0:
                        raise ValueError('Conditional gate requires a positive threshold')
                    best, selected, selected_predictions = tuple(result['rank']), candidate, predictions
                    spec = dict(format='roguard-conditional-contextual-experiment-v1', language='ro',
                        agreement_composition_sha256=digest(root/'scope-agreement-154/actual/ro-composition.json'),
                        selection=candidate, head=dict(language='ro', mode=mode, couplings=coupling[family],
                            checkpoint_sha256=digest(checkpoint)), threshold=result['threshold'])
                    (output/'ro-composition.json').write_text(json.dumps(spec, indent=2)+'\n')
        print(family, json.dumps(selected), flush=True)
    for r in selected_predictions:
        r['predicted'] = int(r['score'] >= selected['frontier']['threshold'])
    (output/'ro-dev-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected_predictions))
    report = dict(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), selected=selected,
        comparisons=comparisons, development_passes=selected['frontier']['counts_pass'],
        baseline=baseline, contextual_calls=int(mask.sum()), total_cases=len(mask),
        calibration_inferred=False, fresh_test_inferred=False, new_foundation_inference=False,
        synthetic_only=True, independently_annotated=False, stable_release_ready=False,
        source_freeze_sha256=digest(study/'source-freeze.json'), parent_freeze_sha256=digest(output/'parent-freeze.json'))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('FINISHED', json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()
