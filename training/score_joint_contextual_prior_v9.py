"""Qualify the frozen joint-selected Ukrainian composite using existing prior features."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from training.diagnose_contextual_complementarity_v9 import read_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features(study, rows):
    import numpy as np
    report = json.loads((study/'actual/report.json').read_text())
    if not report.get('complete'):
        raise ValueError('Incomplete contextual features')
    arrays = []
    offset = 0
    for record in report['chunks']:
        start, stop = record['start'], record['stop']
        if start != offset or not start < stop <= len(rows):
            raise ValueError('Feature chunks incomplete or reordered')
        path = study/'actual'/f'chunk-{start:06d}-features.npz'
        identity = hashlib.sha256(json.dumps([r['token_sha256'] for r in rows[start:stop]]).encode()).hexdigest()
        if (digest(path) != record['feature_sha256'] or identity != record['input_sha256']
                or record['source_freeze_sha256'] != digest(study/'source-freeze.json')):
            raise ValueError('Contextual feature provenance differs')
        value = np.load(path, allow_pickle=False)['features']
        if value.shape != (stop-start, report['feature_width']) or not np.isfinite(value).all():
            raise ValueError('Invalid feature dimensions or values')
        arrays.append(value)
        offset = stop
    if offset != len(rows):
        raise ValueError('Missing final feature chunk')
    return np.concatenate(arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development-only', action='store_true')
    args = parser.parse_args()
    import numpy as np
    from safetensors.numpy import load_file
    from threadpoolctl import threadpool_limits
    from training.gemma4_scope_pipeline_v9 import score
    from training.prepare_instruction_sft_v8 import target
    threadpool_limits(4)
    root = Path('research/v9-stabilization')
    study = root/'joint-contextual-prior-152'
    feature_study = root/'gemma4-qualified-scope-148'
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen candidate source/data differs: ' + path)
    protocol = json.loads((study/'protocol.json').read_text())
    chosen = json.loads((root/'joint-contextual-selection-149/actual/report.json').read_text())['languages']['uk']['selected']
    if chosen['kind'] != 'joint' or not chosen['frontier']['counts_pass']:
        raise ValueError('Joint development prerequisite fails')
    composition = json.loads((root/'joint-contextual-selection-149/actual/uk-composition.json').read_text())
    if composition['selection'] != chosen:
        raise ValueError('Composition selection differs')
    head = composition['head']
    checkpoint = root/chosen['semantic_family']/'actual'/('epoch-'+str(chosen['epoch'])+'.safetensors')
    if digest(checkpoint) != head['checkpoint_sha256']:
        raise ValueError('Semantic checkpoint differs')
    tensors = load_file(checkpoint)
    mixing = composition['mixing']
    gate_path = root/'expanded-scope-gate-147/actual/uk-scope-gate.json.gz'
    if digest(gate_path) != chosen['scope_artifact_sha256']:
        raise ValueError('Scope artifact differs')
    gate = json.loads(gzip.decompress(gate_path.read_bytes()))
    cutoff = protocol['threshold']
    if cutoff != chosen['frontier']['threshold'] or cutoff != composition['threshold']:
        raise ValueError('Frozen threshold differs')
    parent = root/'gemma4-readout-141'
    old = read_rows(parent/'inputs.jsonl.gz')
    features = load_features(parent, old)
    indices = [i for i, r in enumerate(old) if r['partition'] == 'dev' and r['row']['language'] == 'uk']
    dev = [old[i]['row'] for i in indices]
    native = root/chosen['native_family']/'uk'
    def native_values(populations):
        result = {}
        for pop in populations:
            for row in read_rows(native/(pop+'-predictions.jsonl')):
                if row['case_id'] in result:
                    raise ValueError('Duplicate prediction identity')
                result[row['case_id']] = row
        return result
    existing = native_values(('dev', 'scope_dev', 'stress'))
    components = score(features[indices], dev, [existing[r['case_id']]['score'] for r in dev], tensors, head, mixing, gate)
    expected = {r['case_id']: r for r in read_rows(root/'joint-contextual-selection-149/actual/uk-dev-predictions.jsonl')}
    if set(expected) != {r['case_id'] for r in dev}:
        raise ValueError('Development replay population differs')
    error = max(abs(float(v)-expected[r['case_id']]['score']) for r, v in zip(dev, components['score']))
    changed = sum(int(float(v) >= cutoff) != expected[r['case_id']]['predicted'] for r, v in zip(dev, components['score']))
    if error > protocol['full_development_score_tolerance'] or changed:
        raise ValueError(f'Portable development replay differs: error={error}, decisions={changed}')
    replay = dict(complete=True, rows=len(dev), score_max_error=error, decision_changes=changed,
                  new_neural_inference=False, fresh_test_inferred=False,
                  installation_parity=False, source_freeze_sha256=digest(study/'source-freeze.json'))
    (study/'development-replay.json').write_text(json.dumps(replay, indent=2)+'\n')
    print('DEVELOPMENT_REPLAY', json.dumps(replay), flush=True)
    if args.development_only:
        return
    entries = read_rows(feature_study/'inputs.jsonl.gz')
    features = load_features(feature_study, entries)
    rows = [r['row'] for r in entries]
    existing = native_values(('cal', 'exposed'))
    if set(existing) != {r['case_id'] for r in rows}:
        raise ValueError('Prior prediction population differs')
    for row in rows:
        gold = int(row['review']) if 'review' in row else target(row)
        if existing[row['case_id']]['gold'] != gold:
            raise ValueError('Prior reference label differs')
    components = score(features, rows, [existing[r['case_id']]['score'] for r in rows], tensors, head, mixing, gate)
    predictions = [dict(case_id=r['row']['case_id'], language='uk', group=r['group'],
        gold=existing[r['row']['case_id']]['gold'], score=float(value), predicted=int(value >= cutoff)) for r, value in zip(entries, components['score'])]
    metrics = {}
    for group in sorted({r['group'] for r in predictions}):
        group_rows = [r for r in predictions if r['group'] == group]
        positive = sum(r['gold'] for r in group_rows)
        negative = len(group_rows)-positive
        if not positive or not negative:
            raise ValueError('Both classes required in every qualification group')
        missed = sum(r['gold'] and not r['predicted'] for r in group_rows)
        false = sum(not r['gold'] and r['predicted'] for r in group_rows)
        metrics[group] = dict(positive=positive, negative=negative, missed=missed, false_reviews=false,
                              recall=(positive-missed)/positive, false_review_rate=false/negative,
                              passes=missed*100 <= positive and false*20 <= negative)
    output = study/'actual'
    output.mkdir(exist_ok=False)
    (output/'uk-prior-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    report = dict(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), language='uk',
                  selected_threshold=cutoff, metrics=metrics, prior_counts_pass=all(m['passes'] for m in metrics.values()),
                  development_replay=replay, frozen_selection=True, fresh_test_inferred=False,
                  calibration_used_for_selection=False, synthetic_only=True, independently_annotated=False,
                  stable_release_ready=False, installed_execution_verified=False,
                  source_freeze_sha256=digest(study/'source-freeze.json'), feature_report_sha256=digest(feature_study/'actual/report.json'), feature_study=str(feature_study))
    (output/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PRIOR_QUALIFICATION', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
