"""Replay and qualify a frozen cascade without inferring its early exits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.fit_scope_entailment_v9 import metrics
from training.gemma4_structured_v9 import portable_logits
from training.native_fields_v9 import SIZES, policy_mass
from training.prepare_instruction_sft_v8 import target
from training.score_gemma4_qualified_scope_v9 import load_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development-only', action='store_true')
    args = parser.parse_args()
    import numpy as np
    from safetensors.numpy import load_file
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    root = Path('research/v9-stabilization')
    study = root/'conditional-contextual-prior-156'
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen source or input differs: ' + path)
    protocol = json.loads((study/'protocol.json').read_text())
    composition = json.loads((root/'conditional-contextual-155/actual/ro-composition.json').read_text())
    selection = composition['selection']
    if not selection['frontier']['counts_pass'] or selection['kind'] != 'conditional':
        raise ValueError('Development prerequisite fails')
    if digest(root/'scope-agreement-154/actual/ro-composition.json') != composition['agreement_composition_sha256']:
        raise ValueError('Agreement composition differs')
    checkpoint = root/selection['semantic_family']/'actual'/f"epoch-{selection['epoch']}.safetensors"
    if digest(checkpoint) != composition['head']['checkpoint_sha256']:
        raise ValueError('Semantic checkpoint differs')
    cutoff = composition['threshold']
    if cutoff != protocol['threshold'] or cutoff <= 0:
        raise ValueError('Invalid conditional threshold')
    tensors = load_file(checkpoint)
    def semantic(features):
        logits = portable_logits(features, tensors).astype(np.float64)
        def softmax(values):
            exp = np.exp(values-values.max(axis=1, keepdims=True))
            return exp/exp.sum(axis=1, keepdims=True)
        mode = composition['head']['mode']
        if mode == 'binary':
            return softmax(logits[:, -2:])[:, 1]
        if mode not in ('policy', 'coupled_policy'):
            raise ValueError('Unsupported semantic mode')
        fields = [softmax(v) for v in np.split(logits[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)]
        return policy_mass(fields, composition['head']['couplings'] if mode == 'coupled_policy' else None)
    parent = root/'gemma4-readout-141'
    old = read_rows(parent/'inputs.jsonl.gz')
    features = load_features(parent, old)
    dev = {r['case_id']: r for r in read_rows(root/'scope-agreement-154/actual/ro-dev-predictions.jsonl')}
    ix = [i for i, r in enumerate(old) if r['partition'] == 'dev' and r['row']['language'] == 'ro' and dev[r['row']['case_id']]['predicted']]
    values = semantic(features[ix])
    lookup = {old[i]['row']['case_id']: float(v) for i, v in zip(ix, values)}
    expected = read_rows(root/'conditional-contextual-155/actual/ro-dev-predictions.jsonl')
    if {r['case_id'] for r in expected} != set(dev):
        raise ValueError('Development population differs')
    error = max(abs(lookup.get(r['case_id'], 0.)-r['score']) for r in expected)
    changes = sum(int(lookup.get(r['case_id'], 0.) >= cutoff) != r['predicted'] for r in expected)
    if changes or error > protocol['full_development_score_tolerance']:
        raise ValueError('Conditional development replay differs')
    replay = dict(complete=True, cases=len(expected), contextual_calls=len(ix), score_max_error=error,
        decision_changes=changes, installation_parity=False, new_foundation_inference=False,
        fresh_test_inferred=False, source_freeze_sha256=digest(study/'source-freeze.json'))
    (study/'development-replay.json').write_text(json.dumps(replay, indent=2)+'\n')
    print('DEVELOPMENT_REPLAY', json.dumps(replay), flush=True)
    if args.development_only:
        return
    entries = read_rows(study/'inputs.jsonl.gz')
    features = load_features(study, entries)
    values = semantic(features)
    prior = read_rows(root/'scope-agreement-154/actual/ro-prior-predictions.jsonl')
    lookup = {r['row']['case_id']: float(v) for r, v in zip(entries, values)}
    if set(lookup) != {r['case_id'] for r in prior if r['predicted']}:
        raise ValueError('Conditional inference population differs')
    references = {r['case_id']: r for r in prior}
    for r in entries:
        row = r['row']
        gold = int(row['review']) if 'review' in row else target(row)
        if references[row['case_id']]['gold'] != gold or references[row['case_id']]['group'] != r['group']:
            raise ValueError('Prior label or group differs')
    predictions = [dict(r, score=lookup.get(r['case_id'], 0.), predicted=int(lookup.get(r['case_id'], 0.) >= cutoff)) for r in prior]
    actual_metrics = metrics(predictions)
    output = study/'actual'
    (output/'ro-prior-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    report = dict(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), language='ro',
        metrics=actual_metrics, prior_counts_pass=all(m['passes'] for m in actual_metrics.values()),
        development_replay=replay, total_cases=len(prior), contextual_calls=len(entries),
        selected_threshold=cutoff, frozen_selection=True, calibration_used_for_selection=False,
        synthetic_only=True, independently_annotated=False, fresh_test_inferred=False,
        installed_execution_verified=False, stable_release_ready=False,
        source_freeze_sha256=digest(study/'source-freeze.json'), feature_report_sha256=digest(output/'report.json'))
    (output/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PRIOR_QUALIFICATION', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
