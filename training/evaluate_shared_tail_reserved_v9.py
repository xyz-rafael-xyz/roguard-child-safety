"""First neural use of the reserved117 test, gated on frozen installed evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from training.verify_shared_tail_installed_v9 import digest, installed_files

TESTS = {
    'ro': 'ff07b2e373969e0509769de7e3a907a3bf57952467ced6e153ff4abe6e0800f1',
    'uk': 'c4ea1c1b0e45a519654e7dcdf9217f9f55eb7b4315315dcfafcbc084cc8adef6',
}
PREVIOUS = ('paired-installed-118', 'paired-nli-installed-120', 'policy-entailment-installed-122',
    'reason-entailment-installed-124', 'native-coverage-installed-126', 'native-morphology-installed-129',
    'native-scope-operators-installed-132', 'native-full-scope-installed-134', 'native-conversational-installed-138')


def require_installed_gate(report, wheel_hash, bundle_hash):
    if (report.get('complete') is not True or report.get('installed_execution_verified') is not True
            or report.get('prior_counts_pass') is not True or report.get('fresh_test_inferred') is not False
            or report.get('wheel_sha256') != wheel_hash or report.get('bundle_sha256') != bundle_hash
            or set(report.get('languages', {})) != {'ro', 'uk'}):
        raise ValueError('Complete installed bilingual qualification is required before reserved access')
    for language in ('ro', 'uk'):
        item = report['languages'][language]
        replay = item.get('replay', {})
        if (replay.get('decision_changes') != 0 or type(replay.get('score_max_error')) not in (int, float)
                or not 0 <= replay['score_max_error'] <= 1e-5
                or set(item.get('metrics', {})) != {'formal', 'conversational', 'exposed'}):
            raise ValueError('Installed replay or required groups failed')
        for value in item['metrics'].values():
            numbers = [value.get(k) for k in ('positive', 'negative', 'missed', 'false_reviews')]
            if any(type(v) is not int for v in numbers):
                raise ValueError('Invalid installed accuracy counts')
            positive, negative, missed, false = numbers
            if (positive <= 0 or negative <= 0 or not 0 <= missed <= positive or not 0 <= false <= negative
                    or missed*100 > positive or false*20 > negative):
                raise ValueError('Installed accuracy gates failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--installed', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    prior = root/'shared-tail-installed-167'
    report_path = prior/'actual/report.json'
    report = json.loads(report_path.read_text())
    wheel_hash, bundle_hash = digest(args.wheel), digest(args.bundle)
    require_installed_gate(report, wheel_hash, bundle_hash)
    for path, expected in json.loads((prior/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Installed qualification source changed')
    if report['source_freeze_sha256'] != digest(prior/'source-freeze.json') or report['protocol_sha256'] != digest(prior/'protocol.json'):
        raise ValueError('Installed qualification binding changed')
    protocol = json.loads((prior/'protocol.json').read_text())
    if installed_files(args.wheel, args.installed) != protocol['installed_files']:
        raise ValueError('Installed runtime differs from qualified wheel')
    from roguard.entailment_bundle import load_bundle
    from roguard.native_review import prepare_input
    from training.shared_tail_qualification_v9 import candidate_bindings
    from training.stage_shared_tail_bundle_v9 import qualification
    from training.verify_candidate_pipeline_v9 import counts
    specs, bindings = candidate_bindings(root/'shared-tail-fit-165')
    bindings.update(qualification(root))
    staged = json.loads((args.bundle.parent/'staging-evidence.json').read_text())
    if staged['parents'] != bindings or staged['bundle_sha256'] != bundle_hash:
        raise ValueError('Bundle candidate identity changed')
    installed_files(args.wheel, args.installed)
    if any((root/name/'actual/test-access.json').exists() for name in PREVIOUS):
        raise ValueError('Reserved117 test has already been opened by another candidate')
    ledger = root/'paired-scope-117/neural-access-ledger.json'
    if ledger.exists():
        raise ValueError('Reserved117 test already has a neural-access ledger')
    study = root/'shared-tail-reserved-168'
    study.mkdir(exist_ok=False)
    bindings.update({str(path): digest(path) for path in (report_path, prior/'source-freeze.json', prior/'protocol.json',
        Path('training/evaluate_shared_tail_reserved_v9.py'), Path('training/verify_candidate_pipeline_v9.py'), Path('training/verify_shared_tail_installed_v9.py'))})
    (study/'source-freeze.json').write_text(json.dumps(bindings, indent=2)+'\n')
    selection = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), wheel_sha256=wheel_hash,
        bundle_sha256=bundle_hash, installed_report_sha256=digest(report_path),
        source_freeze_sha256=digest(study/'source-freeze.json'), input_sha256=TESTS,
        first_neural_inference_for_this_frozen_test=True, synthetic_only=True, independently_annotated=False,
        analyst_blind=False, independent_observations=False, stable_release_ready=False)
    (study/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    output = study/'actual'
    output.mkdir()
    # Atomic global first-access record precedes opening cases and running models.
    with ledger.open('x') as stream:
        stream.write(json.dumps(dict(study=168, selection_sha256=digest(study/'selection.json'),
            input_sha256=TESTS, started_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
    (output/'test-access.json').write_bytes(ledger.read_bytes())
    inputs = {}
    for language, expected in TESTS.items():
        path = root/'paired-scope-117/data'/f'{language}-test.jsonl'
        if digest(path) != expected:
            raise ValueError('Reserved input bytes changed')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if (len(rows) != 300 or len({r['case_id'] for r in rows}) != 300
                or any(r['language'] != language or type(r['review']) is not int or r['review'] not in (0, 1) for r in rows)
                or sum(r['review'] for r in rows) != 150):
            raise ValueError('Reserved population identity or strata differ')
        inputs[language] = rows
    model = load_bundle(args.bundle, expected_sha256=bundle_hash)
    result = dict(complete=False, languages={}, selection_sha256=digest(study/'selection.json'),
        first_neural_inference=True, synthetic_only=True, independently_annotated=False,
        analyst_blind=False, independent_observations=False, stable_release_ready=False)
    for language, rows in inputs.items():
        values = []
        started = time.monotonic()
        for offset in range(0, len(rows), 4):
            batch = rows[offset:offset+4]
            payloads = [dict(language=language, turns=r['turns']) for r in batch]
            outputs = model.predict_many(payloads, focuses=[r['focus'] for r in batch])
            if len(outputs) != len(batch):
                raise ValueError('Incomplete reserved inference')
            for row, payload, value in zip(batch, payloads, outputs):
                _, focus, identity = prepare_input(payload, row['focus'])
                if value['input_digest'] != identity or value['focus'] != focus or value['language'] != language:
                    raise ValueError('Reserved output binding differs')
                values.append(dict(value, case_id=row['case_id'], gold=row['review']))
            (output/f'{language}-test-predictions.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
            if offset % 100 == 0:
                print('RESERVED', language, len(values), '/', len(rows), flush=True)
        measured = counts(values)
        result['languages'][language] = dict(counts=measured, input_sha256=TESTS[language], seconds=time.monotonic()-started)
        (output/'progress.json').write_text(json.dumps(result, indent=2)+'\n')
        print('RESULT', language, json.dumps(measured), flush=True)
    result.update(complete=True, counts_pass=all(v['counts']['passes'] for v in result['languages'].values()),
        finished_utc=datetime.now(timezone.utc).isoformat())
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
