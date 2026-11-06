"""Reproduce probability cancellation and check unchanged legacy calibration."""
import argparse
import json
from pathlib import Path

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl
from roguard.trajectory_evaluation import checkpoint_id
from roguard.trajectory_sets import WORLDS, calibrate, predict_set


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--study', type=Path, required=True)
    a = p.parse_args()
    if a.study.exists():
        raise ValueError('Refusing to overwrite numerical calibration audit')
    root = Path('research/v10-evidence'); a.study.mkdir(parents=True)
    cases = {name: read_jsonl(root/'data'/(name+'.jsonl')) for name in ('calibration', 'dev', 'challenge', 'v9-regression')}
    lookups = {name: {checkpoint_id(c, point): (c, point) for c in values for point in c['checkpoints']} for name, values in cases.items()}
    report = {'complete': False, 'source_sha256': file_sha(Path(__file__)),
        'runtime_source_sha256': file_sha('src/roguard/trajectory_sets.py'),
        'amendment_sha256': file_sha(root/'protocol-amendment-37.json'), 'historical': {},
        'independent_validation': False, 'stable_release_ready': False}
    for name in ('neural-04', 'uniform-memory-09'):
        source = root/name; old = json.loads((source/'calibration.json').read_text()); rows = []
        for prediction in read_jsonl(source/'calibration-raw.jsonl'):
            case, point = lookups['calibration'][prediction['checkpoint_id']]
            if prediction['pipeline_digest'] != old['pipeline_digest']:
                raise ValueError('Historical pipeline binding differs')
            rows.append({'case_id': prediction['checkpoint_id'], 'split': 'calibration',
                'language': case['language'], 'family': case['family'], 'world': point['world'],
                'pipeline_digest': old['pipeline_digest'], 'probabilities': prediction['probabilities']})
        restored = calibrate(rows, pipeline_digest=old['pipeline_digest'], alpha=old['alpha'])
        if restored != old:
            raise ValueError('Legacy default calibration changed')
        stable = calibrate(rows, pipeline_digest=old['pipeline_digest'], alpha=old['alpha'], method='probability_floor')
        (a.study/(name+'-probability-floor.json')).write_text(json.dumps(stable, indent=2)+'\n')
        populations = {}; inputs = {'calibration': file_sha(source/'calibration-raw.jsonl'), 'old_artifact': file_sha(source/'calibration.json')}
        for population in ('dev', 'challenge', 'v9-regression'):
            path = source/(population+'-raw.jsonl'); inputs[population] = file_sha(path)
            groups = {lang: {'rows': 0, 'changed_sets': 0, 'legacy_missing_gold': 0,
                             'floor_missing_gold': 0, 'legacy_total_worlds': 0, 'floor_total_worlds': 0} for lang in ('ro', 'uk')}
            for prediction in read_jsonl(path):
                case, point = lookups[population][prediction['checkpoint_id']]
                args = {'language': prediction['language'], 'pipeline_digest': old['pipeline_digest'], 'supported': prediction['predicted'] is not None}
                left = predict_set(prediction['probabilities'], old, **args)['hypotheses']
                right = predict_set(prediction['probabilities'], stable, **args)['hypotheses']
                group = groups[case['language']]; group['rows'] += 1
                group['changed_sets'] += left != right
                group['legacy_missing_gold'] += point['world'] not in left
                group['floor_missing_gold'] += point['world'] not in right
                group['legacy_total_worlds'] += len(left); group['floor_total_worlds'] += len(right)
            populations[population] = groups
        report['historical'][name] = {'legacy_artifact_exact': True, 'inputs': inputs,
            'probability_floors': stable['probability_floors'], 'populations': populations,
            'calibration_status': 'Previously exposed engineering data, including known prior-training overlap; not qualification'}
    values = [0.]*432; values[0] = 1e-30; values[1] = 1e-31; values[2] = 1.
    pipeline = 'a'*64
    rows = [{'case_id': lang+str(i), 'language': lang, 'family': str(i), 'split': 'calibration',
             'pipeline_digest': pipeline, 'world': WORLDS[0], 'probabilities': values} for lang in ('ro', 'uk') for i in range(9)]
    old = calibrate(rows, pipeline_digest=pipeline); stable = calibrate(rows, pipeline_digest=pipeline, method='probability_floor')
    left = predict_set(values, old, language='ro', pipeline_digest=pipeline)['hypotheses']
    right = predict_set(values, stable, language='ro', pipeline_digest=pipeline)['hypotheses']
    if len(left) != 432 or right != [WORLDS[0], WORLDS[2]]:
        raise ValueError('Tiny-probability numerical regression')
    report['constructed_numerical_case'] = {'legacy_set_size': len(left), 'floor_set_size': len(right),
        'gold_retained_in_both': WORLDS[0] in left and WORLDS[0] in right, 'floor': stable['probability_floors']['ro'],
        'empirical_model_accuracy_result': False}
    report['complete'] = True
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
