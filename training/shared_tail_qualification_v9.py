"""Strict candidate bindings and portable scores for adapter qualification."""
import json
from pathlib import Path

from training.extract_gemma4_qualified_scope_v9 import digest
from training.native_fields_v9 import SIZES, policy_mass


def candidate_bindings(study):
    """Require completed bilingual development and bind every selected weight."""
    study = Path(study)
    for name, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(name) != expected:
            raise ValueError('Candidate training source differs: ' + name)
    report = json.loads((study/'actual/report.json').read_text())
    if (report.get('complete') is not True or report.get('development_passes') is not True
            or set(report.get('selected', {})) != {'ro', 'uk'}
            or report.get('source_freeze_sha256') != digest(study/'source-freeze.json')):
        raise ValueError('Completed bilingual development qualification is required')
    paths = [study/'source-freeze.json', study/'protocol.json', study/'actual/report.json']
    specs = {}
    for language in ('ro', 'uk'):
        chosen = report['selected'][language]
        if chosen['frontier'].get('counts_pass') is not True:
            raise ValueError('Every selected language must pass its development gates')
        path = study/'actual'/f'{language}-review.json'
        spec = json.loads(path.read_text())
        checkpoint = study/'actual'/f'{language}-selected.safetensors'
        if (spec.get('format') != 'roguard-shared-tail-adapter-experiment-v1'
                or spec.get('language') != language
                or spec.get('source_freeze_sha256') != digest(study/'source-freeze.json')
                or spec.get('checkpoint_sha256') != digest(checkpoint)
                or spec['checkpoint_sha256'] != chosen['checkpoint_sha256']
                or spec.get('threshold') != chosen['frontier']['threshold']
                or spec.get('mode') != chosen['mode']):
            raise ValueError('Selected candidate specification or checkpoint differs')
        specs[language] = spec
        paths += [path, checkpoint, study/'actual'/f'{language}-dev-predictions.jsonl']
    return specs, {str(path): digest(path) for path in paths}


def scores_from_logits(logits, spec):
    import numpy as np
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not len(values) or values.shape[1] != sum(SIZES)+2 or not np.isfinite(values).all():
        raise ValueError('Expected finite semantic and routing logits')
    def softmax(v):
        p = np.exp(v-v.max(axis=1, keepdims=True))
        return p/p.sum(axis=1, keepdims=True)
    if spec['mode'] == 'binary':
        result = softmax(values[:, -2:])[:, 1]
    elif spec['mode'] in ('policy', 'coupled_policy'):
        factors = [softmax(v) for v in np.split(values[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)]
        result = policy_mass(factors, spec['couplings'] if spec['mode'] == 'coupled_policy' else None)
    else:
        raise ValueError('Unknown selected scoring mode')
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError('Invalid selected review score')
    return result


def compare_replay(actual, expected, cutoff, tolerance):
    """Bind case identities and labels before checking score and decision parity."""
    import math
    if not actual or len(actual) != len(expected):
        raise ValueError('Replay population differs')
    lookup = {r['case_id']: r for r in expected}
    if len(lookup) != len(expected) or len({r['case_id'] for r in actual}) != len(actual) or set(lookup) != {r['case_id'] for r in actual}:
        raise ValueError('Replay identities differ')
    error, changes = 0., 0
    for row in actual:
        reference = lookup[row['case_id']]
        if any(row[k] != reference[k] for k in ('language', 'group', 'gold')):
            raise ValueError('Replay group, language or label differs')
        if not math.isfinite(row['score']) or not math.isfinite(reference['score']):
            raise ValueError('Nonfinite replay score')
        error = max(error, abs(row['score']-reference['score']))
        if reference['predicted'] != int(reference['score'] >= cutoff):
            raise ValueError('Reference prediction differs from frozen cutoff')
        changes += int(row['score'] >= cutoff) != reference['predicted']
    if changes or error > tolerance:
        raise ValueError('Selected candidate replay differs')
    return dict(cases=len(actual), score_max_error=error, decision_changes=changes)
