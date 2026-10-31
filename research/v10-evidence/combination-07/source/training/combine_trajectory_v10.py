"""Registered, family-cross-validated score combination on exposed development.

The development partition becomes meta-training. Exposed regression results are
reported only after selection; no independent qualification is claimed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from roguard.assurance import assure, digest
from roguard.trajectory_data import file_sha, prefix, read_jsonl, write_jsonl
from roguard.trajectory_evaluation import checkpoint_id, evaluate
from roguard.trajectory_model import TrajectoryModel


def logit(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Expected a finite probability')
    value = min(1 - 1e-12, max(1e-12, value))
    return math.log(value) - math.log1p(-value)


def fit(x, y, strength):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    prior = np.array([1., 0., 0.])

    def objective(weights):
        z = x @ weights
        value = np.logaddexp(0, z).sum() - y @ z + strength * ((weights - prior) ** 2).sum() / 2
        gradient = x.T @ (expit(z) - y) + strength * (weights - prior)
        return value, gradient

    result = minimize(objective, prior, jac=True, method='L-BFGS-B',
                      bounds=[(0, None), (0, None), (None, None)], options={'maxiter': 1000})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('Combination fit did not converge: ' + result.message)
    return result.x.tolist()


def decision(row, weights):
    margin = sum(a * b for a, b in zip(row['features'], weights))
    return int(margin >= 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--neural', type=Path, required=True)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite a combination study')
    args.output.mkdir(parents=True)
    student = TrajectoryModel(args.student)
    thresholds = {lang: json.loads((args.neural / 'v9-binding' / (lang + '-review.json')).read_text())['threshold']
                  for lang in ('ro', 'uk')}
    bindings = {'student_sha256': student.identity,
                'source_sha256': file_sha(Path(__file__)),
                'amendment_sha256': file_sha(Path('research/v10-evidence/protocol-amendment-04.json')),
                'inputs': {name: {'cases': file_sha(args.data / (name + '.jsonl')),
                                 'neural_predictions': file_sha(args.neural / (name + '-predictions.jsonl'))}
                           for name in ('dev', 'challenge', 'v9-regression')},
                'independent_evaluation': False, 'stable_release_ready': False}
    (args.output / 'protocol.json').write_text(json.dumps(bindings, indent=2) + '\n')

    def rows_for(name):
        cases = read_jsonl(args.data / (name + '.jsonl'))
        neural = {r['checkpoint_id']: r for r in read_jsonl(args.neural / (name + '-predictions.jsonl'))}
        rows = []
        for case in cases:
            for point in case['checkpoints']:
                key = checkpoint_id(case, point)
                n = neural.pop(key)
                s = student.predict(prefix(case, point['focus']))
                if n['input_digest'] != s['input_digest'] or n['language'] != s['language']:
                    raise ValueError('Prediction input binding mismatch')
                if n['predicted'] is None or s['predicted'] is None:
                    raise ValueError('This registered study requires complete component predictions')
                rows.append({'checkpoint_id': key, 'family': case['family'], 'language': case['language'],
                             'gold': point['review'], 'neural': n, 'student_score': s['score'],
                             'features': [logit(n['score']) - logit(thresholds[case['language']]),
                                          logit(s['score']) - logit(student.threshold), 1.]})
        if neural:
            raise ValueError('Unexpected neural predictions')
        return cases, rows

    dev_cases, dev_rows = rows_for('dev')
    settings = {}; selection = {}; out_of_family = {}
    for lang in ('ro', 'uk'):
        rows = [r for r in dev_rows if r['language'] == lang]
        trials = []
        for strength in (1, 10, 100):
            predictions = {}
            for family in sorted({r['family'] for r in rows}):
                training = [r for r in rows if r['family'] != family]
                weights = fit([r['features'] for r in training], [r['gold'] for r in training], strength)
                for row in rows:
                    if row['family'] == family:
                        predictions[row['checkpoint_id']] = decision(row, weights)
            errors = sum(predictions[r['checkpoint_id']] != r['gold'] for r in rows)
            trials.append({'strength': strength, 'errors': errors, 'predictions': predictions})
        best = min(trials, key=lambda r: (r['errors'], -r['strength']))
        out_of_family.update(best['predictions'])
        settings[lang] = fit([r['features'] for r in rows], [r['gold'] for r in rows], best['strength'])
        selection[lang] = {'selected_strength': best['strength'],
                           'trials': [{k: v for k, v in t.items() if k != 'predictions'} for t in trials],
                           'neural_errors': sum(r['neural']['predicted'] != r['gold'] for r in rows)}
    artifact = {'format': 'roguard-score-combination-research1', 'bindings': bindings,
                'weights': settings, 'selection': selection, 'thresholds': thresholds,
                'student_threshold': student.threshold, 'stable_release_ready': False}
    identity = digest(artifact)
    (args.output / 'combination.json').write_text(json.dumps(artifact, indent=2) + '\n')
    reports = {}
    for name in ('dev-out-of-family', 'dev-fitted', 'challenge', 'v9-regression'):
        cases, rows = (dev_cases, dev_rows) if name.startswith('dev') else rows_for(name)
        predictions = []
        for row in rows:
            result = deepcopy(row['neural'])
            result['neural_predicted'] = result['predicted']
            result['predicted'] = (out_of_family[row['checkpoint_id']] if name == 'dev-out-of-family'
                                   else decision(row, settings[row['language']]))
            result['component_neural_score'] = result.pop('score')
            result['component_student_score'] = row['student_score']
            result['combination_identity'] = identity
            result['latency_seconds'] = None  # Cached components do not measure end-to-end latency.
            result['assurance'] = assure({'language': row['language'], 'context_digest': result['input_digest'],
                'hypotheses': result['interpretation_set']['hypotheses'],
                'model_digests': [result['model_identity'], student.identity, identity],
                'review_floor': result['predicted'] == 1})
            predictions.append(result)
        write_jsonl(args.output / (name + '-predictions.jsonl'), predictions)
        reports[name] = evaluate(cases, predictions)
    report = {'selection': selection, 'reports': reports, 'stable_release_ready': False,
              'development_is_meta_training': True,
              'cross_validation_note': 'Used to choose regularization; not an unbiased final evaluation.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: {lang: group['strict_accuracy'] for lang, group in r['groups'].items()}
                      for name, r in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
