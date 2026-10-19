"""Select an existing semantic checkpoint for its frozen combined-system behavior."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from training.budgeted_native_frontier_v9 import frontier
from training.diagnose_contextual_complementarity_v9 import blend, read_rows
from training.gemma4_structured_v9 import portable_logits
from training.native_fields_v9 import SIZES, policy_mass, fit_couplings
from training.prepare_instruction_sft_v8 import target
from training.score_gemma4_qualified_scope_v9 import load_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    from safetensors.numpy import load_file
    from threadpoolctl import threadpool_limits
    from training.score_scope_gate_v9 import score as scope_score
    threadpool_limits(4)
    root = Path('research/v9-stabilization')
    study = root/'joint-contextual-selection-149'
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen source or input differs: ' + path)
    for name, expected in json.loads((study/'software.json').read_text()).items():
        if version(name) != expected:
            raise ValueError('Dependency differs: ' + name)
    protocol = json.loads((study/'protocol.json').read_text())
    parent = root/'gemma4-readout-141'
    expanded = root/'gemma4-structured-expanded-145'
    expanded_report = json.loads((expanded/'actual/report.json').read_text())
    if not expanded_report.get('complete'):
        raise ValueError('Expanded training must finish first')
    output = study/'actual'
    output.mkdir(exist_ok=False)
    if expanded_report.get('development_passes') or expanded_report.get('skipped'):
        (output/'report.json').write_text(json.dumps(dict(complete=True, skipped=True,
            reason='Prior classifier already qualifies for remaining checks'), indent=2)+'\n')
        return
    old = read_rows(parent/'inputs.jsonl.gz')
    features = load_features(parent, old)
    training = [r['row'] for r in old if r['partition'] == 'train']
    additions = [r['row'] for r in read_rows(root/'gemma4-expansion-144/inputs.jsonl.gz')]
    families = {'gemma4-structured-143': training, 'gemma4-structured-expanded-145': training+additions}
    couplings = {}
    parent_hashes = {}
    for family, rows in families.items():
        groups = [(r['language'], target(r)) for r in rows]
        counts = Counter(groups)
        couplings[family] = fit_couplings(rows, [len(rows)/(4*counts[g]) for g in groups])
        path = root/family/'actual/report.json'
        report = json.loads(path.read_text())
        if not report.get('complete'):
            raise ValueError('Incomplete semantic training')
        parent_hashes[str(path)] = digest(path)
        for epoch in protocol['epochs']:
            path = root/family/'actual'/f'epoch-{epoch}.safetensors'
            expected = next(r['checkpoint_sha256'] for r in report['epochs'] if r['epoch'] == epoch)
            if digest(path) != expected:
                raise ValueError('Semantic checkpoint differs')
            parent_hashes[str(path)] = expected
    native_predictions = {}
    native_inputs = {}
    native_checkpoints = {}
    for family in protocol['native_families']:
        for language in ('ro', 'uk'):
            base = root/family/language
            record = json.loads((base/'development.json').read_text())
            if not record.get('complete'):
                raise ValueError('Native training must finish before joint selection')
            parent_hashes[str(base/'development.json')] = digest(base/'development.json')
            spec = json.loads(gzip.decompress((base/'review.json.gz').read_bytes()))
            native_checkpoints[(family, language)] = spec['checkpoint_sha256']
            lookup = {}
            inputs = {}
            for split in ('dev', 'scope_dev')+(('stress',) if language == 'uk' else ()):
                for filename in (split+'.jsonl.gz', split+'-predictions.jsonl'):
                    parent_hashes[str(base/filename)] = digest(base/filename)
                for row in read_rows(base/(split+'-predictions.jsonl')):
                    if row['case_id'] in lookup:
                        raise ValueError('Duplicate native prediction identity')
                    lookup[row['case_id']] = row
                inputs.update({r['case_id']: r for r in read_rows(base/(split+'.jsonl.gz'))})
            native_predictions[(family, language)] = lookup
            native_inputs[(family, language)] = inputs
    # Bind completed parents before comparing any candidate score.
    (output/'parent-freeze.json').write_text(json.dumps(parent_hashes, indent=2)+'\n')
    original = json.loads((root/'expanded-scope-gate-147/actual/report.json').read_text())
    report = dict(complete=False, languages={}, source_freeze_sha256=digest(study/'source-freeze.json'),
                  parent_freeze_sha256=digest(output/'parent-freeze.json'), new_foundation_inference=False,
                  calibration_inferred=False, fresh_test_inferred=False, synthetic_only=True,
                  independently_annotated=False, stable_release_ready=False)
    for language in ('ro', 'uk'):
        ix = [i for i, r in enumerate(old) if r['partition'] == 'dev' and r['row']['language'] == language]
        entries = [old[i] for i in ix]
        rows = [r['row'] for r in entries]
        gold = [target(r) for r in rows]
        base_result = original['languages'][language]['selected']
        best = tuple(base_result['frontier']['rank'])
        selected = dict(kind='retained147', parent_selection=base_result, frontier=base_result['frontier'])
        old_predictions = root/'expanded-scope-gate-147/actual'/(language+'-dev-predictions.jsonl')
        (output/(language+'-dev-predictions.jsonl')).write_bytes(old_predictions.read_bytes())
        gate_path = root/'expanded-scope-gate-147/actual'/(language+'-scope-gate.json.gz')
        if digest(gate_path) != base_result['artifact_sha256']:
            raise ValueError('Frozen scope gate differs')
        gate = json.loads(gzip.decompress(gate_path.read_bytes()))
        multiplier = scope_score(gate, rows, np.ones(len(rows)))
        weight = protocol['native_weights'][language]
        bases = {}
        seen_checkpoints = set()
        for family in protocol['native_families']:
            checkpoint = native_checkpoints[(family, language)]
            if checkpoint in seen_checkpoints:
                continue
            seen_checkpoints.add(checkpoint)
            lookup = native_predictions[(family, language)]
            inputs = native_inputs[(family, language)]
            if set(lookup) != {r['case_id'] for r in rows}:
                raise ValueError('Native development population differs')
            if any(inputs[r['case_id']] != r or lookup[r['case_id']]['gold'] != y for r, y in zip(rows, gold)):
                raise ValueError('Native input content or label differs')
            bases[family] = [lookup[r['case_id']]['score'] for r in rows]
        comparisons = []
        for family in families:
            for epoch in protocol['epochs']:
                checkpoint = root/family/'actual'/f'epoch-{epoch}.safetensors'
                logits = portable_logits(features[ix], load_file(checkpoint)).astype(np.float64)
                def softmax(value):
                    exp = np.exp(value-value.max(axis=1, keepdims=True))
                    return exp/exp.sum(axis=1, keepdims=True)
                fields = [softmax(v) for v in np.split(logits[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)]
                variants = dict(binary=softmax(logits[:, -2:])[:, 1], policy=policy_mass(fields),
                                coupled_policy=policy_mass(fields, couplings[family]))
                for mode, semantic in variants.items():
                    for native_family, native in bases.items():
                        mixed = np.array([blend(float(n), float(s), weight, 'logit') for n, s in zip(native, semantic)])
                        values = mixed*multiplier
                        population = [dict(gold=y, group=r['group'], score=float(v)) for r, y, v in zip(entries, gold, values)]
                        result = frontier(population)
                        selection = dict(kind='joint', semantic_family=family, epoch=epoch, semantic_mode=mode,
                            native_family=native_family, native_checkpoint_sha256=native_checkpoints[(native_family, language)],
                            semantic_checkpoint_sha256=digest(checkpoint), scope_artifact_sha256=digest(gate_path),
                            native_weight=weight, scope_power=gate['selected']['power'], frontier=result)
                        comparisons.append(selection)
                        if tuple(result['rank']) < best:
                            best = tuple(result['rank'])
                            selected = selection
                            predictions = [dict(case_id=r['row']['case_id'], language=language, **p,
                                predicted=int(p['score'] >= result['threshold'])) for r, p in zip(entries, population)]
                            (output/(language+'-dev-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
                            spec = dict(selection=selection, head=dict(language=language, mode=mode, couplings=couplings[family],
                                checkpoint_sha256=selection['semantic_checkpoint_sha256']),
                                mixing=dict(mode='logit', native_weight=weight), threshold=result['threshold'])
                            (output/(language+'-composition.json')).write_text(json.dumps(spec, indent=2)+'\n')
                print(language, family, epoch, 'best', json.dumps(selected), flush=True)
        report['languages'][language] = dict(selected=selected, comparisons=comparisons)
        (output/'development-progress.json').write_text(json.dumps(report, indent=2)+'\n')
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                  development_passes=all(r['selected']['frontier']['counts_pass'] for r in report['languages'].values()))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
