"""Replay all memory-wrapper cases, then verify fresh installed integration."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import zipfile

from roguard.assurance import digest
from roguard.committee_trajectory import CommitteeTrajectoryModel
from roguard.trajectory_data import file_sha, input_identity, prefix, read_jsonl, write_jsonl
from roguard.trajectory_evaluation import checkpoint_id, evaluate
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'base', 'selector', 'reference', 'installed_wheel', 'study'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--sha256', required=True); a = p.parse_args()
    if a.study.exists():
        raise ValueError('Refusing to overwrite pipeline verification')
    import roguard.committee_trajectory as module
    installed = Path(module.__file__).resolve()
    if installed == Path('src/roguard/committee_trajectory.py').resolve():
        raise ValueError('Expected an installed package')
    checked = 0
    with zipfile.ZipFile(a.installed_wheel) as wheel:
        for name in wheel.namelist():
            if name.startswith('roguard/') and not name.endswith('/'):
                if (installed.parent.parent/name).read_bytes() != wheel.read(name):
                    raise ValueError('Installed package file differs')
                checked += 1
    previous = json.loads((a.reference/'report.json').read_text()); binding = json.loads((a.reference/'protocol.json').read_text())
    if not previous['complete'] or previous['decision'] != 'BINARY_RUNTIME_ENGINEERING_PASS_ONLY':
        raise ValueError('Expected completed passing committee reference')
    selector = TrajectoryModel(a.selector, expected_sha256=binding['selector_sha256'])
    model = CommitteeTrajectoryModel.from_bundle(a.bundle, expected_sha256=a.sha256, base=a.base, selector=selector)
    root = Path('research/v10-evidence'); paths = {'broad': root/'broad-v9-regression/cases.jsonl',
        **{name: root/'data'/(name+'.jsonl') for name in ('dev', 'challenge', 'v9-regression')}}
    cases = {}; records = {}; lookup = {}; references = {}; fresh_ids = set()
    for name, path in paths.items():
        predictions_path = a.reference/(name+'-predictions.jsonl')
        if file_sha(path) != binding['inputs'][name] or file_sha(predictions_path) != previous['prediction_sha256'][name]:
            raise ValueError('Reference inputs or predictions differ')
        cases[name] = read_jsonl(path)
        records[name] = {checkpoint_id(c, point): (c, point) for c in cases[name] for point in c['checkpoints']}
        rows = read_jsonl(predictions_path); references[name] = {r['checkpoint_id']: r for r in rows}
        if len(references[name]) != len(records[name]) or set(references[name]) != set(records[name]):
            raise ValueError('Incomplete reference population')
        patterns = set()
        for row in rows:
            case, point = records[name][row['checkpoint_id']]
            _, inference = prepare_memory(prefix(case, point['focus']), selector, 1200, preserve_complete=True)
            identity = input_identity(inference); key = (row['language'], identity)
            if identity != row['inference_input_digest'] or (key in lookup and lookup[key] != row['runtime']):
                raise ValueError('Reference memory input or duplicate prediction differs')
            lookup[key] = row['runtime']
            pattern = (row['language'], tuple(v['predicted'] for _, v in sorted(row['runtime']['components'].items())))
            if pattern not in patterns or row['baseline_predicted'] != row['gold']:
                fresh_ids.add((name, row['checkpoint_id'])); patterns.add(pattern)
    a.study.mkdir(parents=True)
    protocol = {'format': 'roguard-committee-trajectory-verification-1', 'source_sha256': file_sha(Path(__file__)),
        'runtime_source_sha256': file_sha(installed), 'reference_report_sha256': file_sha(a.reference/'report.json'),
        'amendment_sha256': file_sha(root/'protocol-amendment-40.json'), 'installed_wheel_sha256': file_sha(a.installed_wheel),
        'installed_package_files_verified': checked, 'model_identity': model.identity, 'pipeline_digest': model.pipeline_digest(),
        'predictor_identity': model.predictor_identity(), 'selector_sha256': selector.identity,
        'bundle_sha256': a.sha256, 'fitting_performed': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'runtime-source.py').write_bytes(installed.read_bytes())
    for lang, component in model.components.items():
        component.predict = lambda payload, lang=lang: deepcopy(lookup[(lang, input_identity(payload))])
    started = time.monotonic(); replay_records = []; fresh_expected = {}; populations = {}
    for name in paths:
        outputs = []
        for cid, (case, point) in records[name].items():
            payload = prefix(case, point['focus']); result = model.predict(payload); expected = references[name][cid]
            if (result['input_digest'] != expected['input_digest'] or result['inference_input_digest'] != expected['inference_input_digest']
                    or result['components'] != expected['runtime']['components'] or result['predicted'] != expected['runtime']['predicted']
                    or result['committee_identity'] != expected['runtime']['model_identity']):
                raise ValueError('Integrated memory or binary output differs from reference')
            outputs.append({'checkpoint_id': cid, **result})
            replay_records.append({'population': name, 'checkpoint_id': cid, 'output_digest': digest(result),
                'predicted': result['predicted'], 'input_digest': result['input_digest'],
                'inference_input_digest': result['inference_input_digest'], 'neural_output_replayed': True})
            if (name, cid) in fresh_ids:
                fresh_expected[(name, cid)] = result
        populations[name] = evaluate(cases[name], outputs)
    if len(replay_records) != 6002 or any(component._engine is not None for component in model.components.values()):
        raise ValueError('Replay count or no-inference contract differs')
    write_jsonl(a.study/'replay-checks.jsonl', replay_records)
    for component in model.components.values():
        del component.predict
    replay_seconds = time.monotonic()-started; fresh = []; tick = time.monotonic()
    for lang in ('ro', 'uk'):
        for (name, cid), expected in fresh_expected.items():
            case, point = records[name][cid]
            if case['language'] != lang:
                continue
            actual = model.predict(prefix(case, point['focus']))
            if actual != expected:
                raise ValueError('Fresh installed pipeline differs from full replay')
            fresh.append({'population': name, 'checkpoint_id': cid, 'output_digest': digest(actual),
                          'all_output_fields_exact': True})
    write_jsonl(a.study/'fresh-checks.jsonl', fresh)
    report = {'complete': True, 'replayed_checkpoints': len(replay_records), 'fresh_neural_inferences': len(fresh),
        'all_fresh_output_fields_exact': True, 'populations': populations,
        'replay_seconds_not_neural_inference': replay_seconds, 'fresh_check_seconds': time.monotonic()-tick,
        'decision': 'BINARY_INTEGRATION_PASS_ONLY' if all(
            g['strict_accuracy'] == 1 for r in populations.values() for g in r['groups'].values()) else 'REJECT',
        'semantic_sets_calibrated': False, 'independent_validation': False, 'stable_release_ready': False}
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'populations'}, indent=2))


if __name__ == '__main__':
    main()
