"""Prior qualification of the frozen recall-first Ukrainian cascade."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.fit_scope_entailment_v9 import metrics, relative_or
from training.gemma4_structured_v9 import portable_logits
from training.native_fields_v9 import SIZES, policy_mass
from training.portable_scope_gate_v9 import ScopeGate
from training.prepare_instruction_sft_v8 import target
from training.score_gemma4_qualified_scope_v9 import load_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--development-only', action='store_true')
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    study = root/'recall-first-contextual-prior-160'
    chosen = root/'recall-first-contextual-159'
    candidate = json.loads((chosen/'actual/report.json').read_text())
    if not candidate['complete'] or not candidate['development_passes'] or candidate['selected']['kind'] != 'conditional':
        raise ValueError('Development prerequisite fails')
    composition = json.loads((chosen/'actual/uk-composition.json').read_text())
    if composition['selection'] != candidate['selected']:
        raise ValueError('Composition differs from development selection')
    first_path = root/'scope-entailment-153/actual/uk-selection.json'
    if digest(first_path) != composition['entailment_selection_sha256']:
        raise ValueError('First-stage selection differs')
    first = json.loads(first_path.read_text())
    checkpoint = root/candidate['selected']['semantic_family']/'actual'/f"epoch-{candidate['selected']['epoch']}.safetensors"
    if digest(checkpoint) != composition['head']['checkpoint_sha256']:
        raise ValueError('Semantic checkpoint differs')
    gate_path = root/'expanded-scope-gate-147/actual/uk-scope-gate.json.gz'
    reason = root/'reason-coverage-123/candidate'
    if digest(gate_path) != first['gate_sha256'] or digest(reason/'review.json') != first['entailment_artifact_sha256']:
        raise ValueError('Reason or scope artifact differs')
    if args.prepare:
        study.mkdir(exist_ok=False)
        protocol = dict(study=160, created_utc=datetime.now(timezone.utc).isoformat(), language='uk',
            candidate='Frozen159 recall-first entailment-scope then contextual check', threshold=composition['threshold'],
            full_development_score_tolerance=1e-5, new_foundation_inference=False,
            features='Previously frozen148 calibration/exposed representations',
            calibration_used_for_selection=False, fresh_test_inferred=False,
            synthetic_only=True, independently_annotated=False, stable_release_ready=False)
        (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
        paths = [study/'protocol.json', chosen/'source-freeze.json', chosen/'actual/report.json',
            chosen/'actual/uk-composition.json', chosen/'actual/uk-dev-predictions.jsonl', first_path,
            checkpoint, gate_path, reason/'review.json', root/'gemma4-readout-141/source-freeze.json',
            root/'gemma4-readout-141/inputs.jsonl.gz', root/'gemma4-readout-141/actual/report.json',
            root/'gemma4-qualified-scope-148/source-freeze.json', root/'gemma4-qualified-scope-148/inputs.jsonl.gz',
            root/'gemma4-qualified-scope-148/actual/report.json']
        paths += [reason/(f'uk_{population}'+suffix) for population in ('dev', 'scope_dev', 'stress', 'cal', 'exposed') for suffix in ('.jsonl.gz', '-predictions.jsonl')]
        paths += [Path('training')/p for p in ('qualify_recall_first_contextual_v9.py', 'diagnose_contextual_complementarity_v9.py',
            'fit_scope_entailment_v9.py', 'gemma4_structured_v9.py', 'native_fields_v9.py', 'portable_scope_gate_v9.py',
            'prepare_instruction_sft_v8.py', 'score_gemma4_qualified_scope_v9.py', 'quote_scope_v9.py')]
        (study/'source-freeze.json').write_text(json.dumps({str(p): digest(p) for p in paths}, indent=2)+'\n')
        print('PREPARED', json.dumps(protocol), flush=True)
        return
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen source or input differs: ' + path)
    import numpy as np
    from safetensors.numpy import load_file
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    tensors = load_file(checkpoint)
    gate = ScopeGate(json.loads(gzip.decompress(gate_path.read_bytes())))
    cutoff = composition['threshold']
    protocol = json.loads((study/'protocol.json').read_text())
    if cutoff <= 0 or cutoff != protocol['threshold']:
        raise ValueError('Invalid conditional threshold')
    def evaluate(feature_study, partition):
        entries = read_rows(feature_study/'inputs.jsonl.gz')
        features = load_features(feature_study, entries)
        ix = [i for i, r in enumerate(entries) if r['row']['language'] == 'uk' and (r['partition'] == 'dev' if partition == 'dev' else True)]
        rows = [entries[i] for i in ix]
        populations = ('dev', 'scope_dev', 'stress') if partition == 'dev' else ('cal', 'exposed')
        reasons, originals = {}, {}
        for population in populations:
            for r in read_rows(reason/(f'uk_{population}-predictions.jsonl')):
                if r['case_id'] in reasons:
                    raise ValueError('Duplicate reason identity')
                reasons[r['case_id']] = r
            originals.update({r['case_id']: r for r in read_rows(reason/(f'uk_{population}.jsonl.gz'))})
        if set(reasons) != {r['row']['case_id'] for r in rows}:
            raise ValueError('Reason population differs')
        active = []
        predictions = []
        for i, entry in enumerate(rows):
            row = entry['row']
            key = row['case_id']
            other = originals[key]
            gold = int(row['review']) if 'review' in row else target(row)
            if row['turns'] != other['turns'] or row.get('focus', len(row['turns'])-1) != other.get('focus', len(other['turns'])-1) or reasons[key]['gold'] != gold:
                raise ValueError('Reason content or label differs')
            scores = reasons[key]
            base = max(scores['support_score'], scores['revision_score']) if first['selected']['mode'] == 'raw_max' else relative_or(scores['support_score'], scores['revision_score'], first['original_reason_cutoffs'])
            value = base*gate.probability(row)**first['selected']['power']
            if value >= composition['first_stage_cutoff']:
                active.append(i)
            predictions.append(dict(case_id=key, language='uk', group=entry['group'], gold=gold, score=0., predicted=0))
        logits = portable_logits(features[[ix[i] for i in active]], tensors).astype(np.float64)
        def softmax(values):
            exp = np.exp(values-values.max(axis=1, keepdims=True))
            return exp/exp.sum(axis=1, keepdims=True)
        mode = composition['head']['mode']
        if mode == 'binary':
            semantic = softmax(logits[:, -2:])[:, 1]
        elif mode in ('policy', 'coupled_policy'):
            fields = [softmax(v) for v in np.split(logits[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)]
            semantic = policy_mass(fields, composition['head']['couplings'] if mode == 'coupled_policy' else None)
        else:
            raise ValueError('Unsupported semantic mode')
        for i, value in zip(active, semantic):
            predictions[i].update(score=float(value), predicted=int(value >= cutoff))
        return predictions, len(active)
    dev, calls = evaluate(root/'gemma4-readout-141', 'dev')
    expected = {r['case_id']: r for r in read_rows(chosen/'actual/uk-dev-predictions.jsonl')}
    if set(expected) != {r['case_id'] for r in dev}:
        raise ValueError('Development population differs')
    error = max(abs(r['score']-expected[r['case_id']]['score']) for r in dev)
    changed = sum(r['predicted'] != expected[r['case_id']]['predicted'] for r in dev)
    if changed or error > protocol['full_development_score_tolerance']:
        raise ValueError('Development replay differs')
    replay = dict(complete=True, rows=len(dev), contextual_calls=calls, score_max_error=error,
        decision_changes=changed, installation_parity=False, fresh_test_inferred=False)
    (study/'development-replay.json').write_text(json.dumps(replay, indent=2)+'\n')
    print('DEVELOPMENT_REPLAY', json.dumps(replay), flush=True)
    if args.development_only:
        return
    prior, calls = evaluate(root/'gemma4-qualified-scope-148', 'prior')
    prior_metrics = metrics(prior)
    output = study/'actual'
    output.mkdir(exist_ok=False)
    (output/'uk-prior-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prior))
    report = dict(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), language='uk',
        metrics=prior_metrics, prior_counts_pass=all(m['passes'] for m in prior_metrics.values()),
        development_replay=replay, total_cases=len(prior), contextual_calls=calls,
        selected_threshold=cutoff, frozen_selection=True, calibration_used_for_selection=False,
        synthetic_only=True, independently_annotated=False, fresh_test_inferred=False,
        new_foundation_inference=False, installed_execution_verified=False, stable_release_ready=False,
        source_freeze_sha256=digest(study/'source-freeze.json'))
    (output/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PRIOR_QUALIFICATION', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
