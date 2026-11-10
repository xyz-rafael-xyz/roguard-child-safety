"""Family-level split-conformal interpretation sets for a frozen pipeline.

The finite-sample rank is conservative when calibration is too small. Coverage
requires exchangeable family maxima and a correct annotation/ontology contract;
it is not an individual, conditional, temporal or real-world safety guarantee.
"""
from __future__ import annotations

import itertools
import math

from .assurance import digest, policy_digest
from .evidence import ONTOLOGY

WORLDS = [dict(zip(ONTOLOGY, values)) for values in itertools.product(*ONTOLOGY.values())]
WORLD_IDS = {digest(w): i for i, w in enumerate(WORLDS)}


def distribution(values):
    if not isinstance(values, list) or len(values) != len(WORLDS):
        raise ValueError('Expected a score for every interpretation')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values)-1) > 1e-6:
        raise ValueError('Invalid interpretation distribution')
    return values


def calibrate(rows, *, pipeline_digest, alpha=0.1, method='one_minus_probability'):
    if method not in ('one_minus_probability', 'probability_floor'):
        raise ValueError('Unknown interpretation calibration method')
    if type(alpha) not in (float, int) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    if not isinstance(pipeline_digest,str) or len(pipeline_digest)!=64 or any(c not in '0123456789abcdef' for c in pipeline_digest):
        raise ValueError('Invalid frozen pipeline identity')
    groups = {'ro': {}, 'uk': {}}
    identities = set()
    for row in rows:
        if row['split'] != 'calibration' or row['language'] not in groups:
            raise ValueError('Calibration accepts only held-out calibration rows')
        if row['pipeline_digest'] != pipeline_digest or row['case_id'] in identities:
            raise ValueError('Mismatched pipeline or duplicate calibration prediction')
        if not isinstance(row['family'],str) or not row['family']:
            raise ValueError('Calibration requires a scenario-family identifier')
        identities.add(row['case_id'])
        world = WORLD_IDS.get(digest(row['world']))
        if world is None:
            raise ValueError('Gold interpretation is outside the ontology')
        values = distribution(row['probabilities'])
        table = groups[row['language']]
        if method == 'probability_floor':
            table[row['family']] = min(table.get(row['family'], 1.0), values[world])
        else:
            score = 1-values[world]
            table[row['family']] = max(table.get(row['family'], 0), score)
    quantiles, counts = {}, {}
    for lang, table in groups.items():
        scores = sorted(table.values(), reverse=method == 'probability_floor')
        n = len(scores)
        rank = math.ceil((n+1)*(1-alpha))
        # Insufficient families include all worlds in either representation.
        quantiles[lang] = scores[rank-1] if rank <= n else (0.0 if method == 'probability_floor' else 1.0)
        counts[lang] = n
    result = {'format': 'roguard-joint-calibration-v10-research1',
              'pipeline_digest': pipeline_digest, 'policy_digest': policy_digest(),
              'alpha': alpha, 'quantiles': quantiles, 'families_per_language': counts,
              'calibration_digest': digest(rows), 'unit': 'maximum_nonconformity_per_family',
              'assumptions': ['exchangeable scenario families', 'same frozen inference pipeline',
                              'correct in-ontology labels', 'no calibration-driven model selection'],
              'real_world_coverage_established': False}
    if method == 'probability_floor':
        result['format'] = 'roguard-joint-calibration-v10-research2'
        result['probability_floors'] = result.pop('quantiles')
        result['unit'] = 'minimum_true_world_probability_per_family'
    result['digest'] = digest(result)
    return result


def predict_set(values, calibration, *, language, pipeline_digest, supported=True):
    distribution(values)
    expected = {k: v for k, v in calibration.items() if k != 'digest'}
    if calibration.get('digest') != digest(expected) or calibration.get('format') not in (
            'roguard-joint-calibration-v10-research1', 'roguard-joint-calibration-v10-research2'):
        raise ValueError('Calibration integrity mismatch')
    if calibration['pipeline_digest'] != pipeline_digest or calibration['policy_digest'] != policy_digest():
        raise ValueError('Calibration belongs to another pipeline or policy')
    if language not in ('ro', 'uk') or type(supported) is not bool:
        raise ValueError('Invalid language or support flag')
    use_floor = calibration['format'] == 'roguard-joint-calibration-v10-research2'
    q = calibration['probability_floors' if use_floor else 'quantiles'][language]
    if type(q) not in (float, int) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('Invalid calibration quantile')
    selected = [w.copy() for w, p in zip(WORLDS, values) if (p >= q if use_floor else 1-p <= q)] if supported else []
    reason = 'CALIBRATED_SET'
    if not supported or not selected:
        selected = [w.copy() for w in WORLDS]
        reason = 'UNSUPPORTED_OR_EMPTY_SET'
    return {'hypotheses': selected, 'reason': reason, 'calibration_digest': calibration['digest'],
            'coverage_assumptions_verified': False, 'outside_ontology_ruled_out': False}
