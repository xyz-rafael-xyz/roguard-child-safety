"""Development-only, grouped cross-validation for native/structured fusion.

No text, source-stratum identifier, case identifier or gold interpretation is a
predictive feature. Calibration is opened only by the separate audit stage.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

from roguard.evidence import ONTOLOGY, disposition
from roguard.semantic import normalize_semantic_text

FIELDS = tuple(ONTOLOGY)
REGULARIZATION = (0.1, 1.0, 10.0, 100.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    data = Path(path).read_bytes()
    if str(path).endswith('.gz'):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.splitlines()]


def features(native, structured):
    score = native['score']
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid native probability')
    if structured['valid_label_tokens'] is not True:
        return None
    probabilities, labels = [], []
    for field in FIELDS:
        values = structured['conditional_field_probabilities'][field]
        if (len(values) != len(ONTOLOGY[field]) or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
                or abs(sum(values)-1) > 1e-5):
            raise ValueError('Invalid structured probability vector')
        label = structured['interpretation'][field]
        if label not in ONTOLOGY[field]:
            raise ValueError('Invalid structured label')
        probabilities.extend(values)
        labels.extend(float(label == item) for item in ONTOLOGY[field])
    bounded = min(1-1e-6, max(1e-6, score))
    return [score, math.log(bounded/(1-bounded))] + probabilities + labels + [score*p for p in probabilities]


def strata(row):
    return 'conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal'


def folds(rows):
    """Keep same-focal and supplied pair groups together using connected sets."""
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        text = normalize_semantic_text(row['turns'][row['focus']]['text']).translate(str.maketrans('şţŞŢ', 'șțȘȚ'))
        keys = [('focal', row['language'], text)]
        if row.get('pair_group'):
            keys.append(('pair', row['language'], row['pair_group']))
        for key in keys:
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row['case_id'])
    assignment = {key: int(hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(), 16) % 5 for key, ids in groups.items()}
    return [assignment[root(i)] for i in range(len(rows))]


def fit(x, y, kinds, regularization):
    import numpy as np
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    groups = list(zip(kinds, y.astype(int).tolist()))
    frequencies = Counter(groups)
    if len(frequencies) != 4:
        raise ValueError('Fit split lacks a formal/conversational class stratum')
    weights = np.array([len(x)/(4*frequencies[g]) for g in groups])
    penalty = np.full(design.shape[1], regularization); penalty[0] = 0
    beta = np.zeros(design.shape[1])
    def objective(value):
        logits = design@value
        return float(np.sum(weights*(np.logaddexp(0, logits)-y*logits))+.5*np.sum(penalty*value*value))
    converged = False
    for iteration in range(100):
        logits = np.clip(design@beta, -700, 700)
        probability = 1/(1+np.exp(-logits))
        gradient = design.T@(weights*(probability-y))+penalty*beta
        if np.max(np.abs(gradient)) < 1e-6:
            converged = True; break
        hessian = design.T@((weights*probability*(1-probability))[:, None]*design)+np.diag(penalty)
        direction = np.linalg.solve(hessian, gradient)
        current = objective(beta); rate = 1.0
        while rate >= 2**-30:
            candidate = beta-rate*direction
            if objective(candidate) <= current-1e-4*rate*float(gradient@direction):
                beta = candidate; break
            rate *= .5
        else:
            raise ValueError('Fusion optimization line search failed')
    if not converged or not np.isfinite(beta).all():
        raise ValueError('Fusion optimizer did not converge')
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=beta.tolist(), iterations=iteration)


def probability(model, vector):
    if vector is None:
        return None
    if len(vector) != len(model['mean']):
        raise ValueError('Fusion feature dimension differs')
    value = model['coefficients'][0]+sum(w*(x-m)/s for w, x, m, s in zip(model['coefficients'][1:], vector, model['mean'], model['scale']))
    return 1/(1+math.exp(-max(-700, min(700, value))))


def metrics(rows, truth, scores, threshold):
    from training.reserved_metrics_v9 import counts
    return {kind: counts([dict(gold=int(y), predicted=None if s is None else int(s >= threshold))
                         for r, y, s in zip(rows, truth, scores) if strata(r) == kind])
            for kind in ('formal', 'conversational')}


def threshold(rows, truth, scores):
    boundaries = []
    for kind in ('formal', 'conversational'):
        positive = [s for r, y, s in zip(rows, truth, scores) if strata(r) == kind and y == 1]
        budget = len(positive)//100
        missing = sum(s is None for s in positive)
        if not positive or missing > budget:
            return 0.0
        valid = sorted(s for s in positive if s is not None)
        boundaries.append(max(0.0, valid[budget-missing]-1e-8))
    return min(boundaries)


def load_split(root, language, split):
    candidate = root/('native-full-22-'+language)
    paths = dict(rows=candidate/(split+'.jsonl.gz'), native=candidate/(split+'-predictions.jsonl'),
                 structured=root/'interpreter-full-21'/split/'predictions.jsonl')
    values = {key: read(path) for key, path in paths.items()}
    structured = {r['case_id']: r for r in values['structured']}
    if len(structured) != len(values['structured']):
        raise ValueError('Duplicate structured case identity')
    rows, native = values['rows'], values['native']
    if len(rows) != len(native) or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Native coverage differs')
    x, y = [], []
    for row, n in zip(rows, native):
        s = structured[row['case_id']]
        gold = int(disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE')
        if (row['case_id'] != n['case_id'] or n['gold'] != s['gold'] or n['gold'] != gold
                or row['language'] != language or n['language'] != language or s['language'] != language):
            raise ValueError('Model outputs do not align')
        x.append(features(n, s)); y.append(n['gold'])
    return rows, x, y, {key: sha(path) for key, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'audit'))
    parser.add_argument('--language', choices=('ro', 'uk'), required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization'); output = root/'review-fusion-36'/args.language
    protocol = root/'review-fusion-36/protocol.json'
    parent_paths = [root/('native-full-22-'+args.language)/'review.json.gz', root/'interpreter-full-21/selection.json']
    parent_hashes = {str(path.relative_to(root)): sha(path) for path in parent_paths}
    if args.stage == 'prepare':
        rows, x, y, hashes = load_split(root, args.language, 'dev')
        assignments = folds(rows)
        if set(assignments) != set(range(5)):
            raise ValueError('Connected development groups do not cover all five folds')
        kinds = [strata(r) for r in rows]; trials = []
        for regularization in REGULARIZATION:
            scores = [None]*len(rows)
            for fold in range(5):
                train = [i for i in range(len(rows)) if assignments[i] != fold and x[i] is not None]
                model = fit([x[i] for i in train], [y[i] for i in train], [kinds[i] for i in train], regularization)
                for i in range(len(rows)):
                    if assignments[i] == fold:
                        scores[i] = probability(model, x[i])
            limit = threshold(rows, y, scores); counts = metrics(rows, y, scores, limit)
            # Unchanged gate eligibility precedes false-review ranking.
            rank = (not all(v['passes'] for v in counts.values()), max(v['false_review_rate'] for v in counts.values()),
                    sum(v['false_review_rate'] for v in counts.values()), -regularization)
            trials.append(dict(regularization=regularization, threshold=limit, metrics=counts, rank=rank, scores=scores))
        selected = min(trials, key=lambda v: v['rank'])
        valid = [i for i in range(len(rows)) if x[i] is not None]
        fitted = fit([x[i] for i in valid], [y[i] for i in valid], [kinds[i] for i in valid], selected['regularization'])
        output.mkdir(parents=True, exist_ok=False)
        record = dict(created_utc=datetime.now(timezone.utc).isoformat(), language=args.language,
            selection_population='Grouped out-of-fold development predictions; base models already used development for selection.',
            calibration_opened=False, reserved_probes_used=False, stable_release_ready=False,
            source_sha256=sha(__file__), protocol_sha256=sha(protocol), development_hashes=hashes, parent_hashes=parent_hashes,
            fold_counts=dict(Counter(assignments)), selected={k: v for k, v in selected.items() if k != 'scores'},
            trials=[{k: v for k, v in trial.items() if k != 'scores'} for trial in trials], model=fitted)
        (output/'selection.json').write_text(json.dumps(record, indent=2)+'\n')
        (output/'oof-predictions.jsonl').write_text(''.join(json.dumps(dict(case_id=r['case_id'], gold=t, score=s,
            fold=f, predicted=None if s is None else int(s >= selected['threshold'])))+'\n' for r,t,s,f in zip(rows,y,selected['scores'],assignments)))
        print(json.dumps(record['selected'], indent=2))
    else:
        selection = output/'selection.json'; record = json.loads(selection.read_text())
        if (record['source_sha256'] != sha(__file__) or record['protocol_sha256'] != sha(protocol)
                or record['parent_hashes'] != parent_hashes):
            raise ValueError('Frozen fusion recipe changed')
        _, _, _, hashes = load_split(root, args.language, 'dev')
        if hashes != record['development_hashes']:
            raise ValueError('Development evidence changed')
        rows, x, y, hashes = load_split(root, args.language, 'cal')
        scores = [probability(record['model'], value) for value in x]
        counts = metrics(rows, y, scores, record['selected']['threshold'])
        result = dict(selection_sha256=sha(selection), calibration_hashes=hashes, calibration=counts,
                      counts_pass=all(v['passes'] for v in counts.values()) and all(v['passes'] for v in record['selected']['metrics'].values()),
                      stable_release_ready=False, reserved_probes_used=False,
                      requirements_remaining=['Installed composite runtime parity', 'Frozen reserved actual-runtime probes', 'Product integration and release checks'])
        with (output/'cal-predictions.jsonl').open('x') as stream:
            stream.write(''.join(json.dumps(dict(case_id=r['case_id'], gold=t, score=s,
                predicted=None if s is None else int(s >= record['selected']['threshold'])))+'\n' for r,t,s in zip(rows,y,scores)))
        (output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
