"""Semantic-only boosted combination excluding the direct binary expert.

Uses the predeclared three-expert features, with no raw-message features.
No text, case identifier or source-group identifier is a predictive feature.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from roguard.evidence import ONTOLOGY, disposition
from training.review_fusion_v9 import (REGULARIZATION, sha, read, features, folds,
    strata, fit, probability, metrics)
from training.recall_gap_threshold_v9 import recall_gap_threshold

FIELD_STUDIES = {'ro': 'semantic-selection-71', 'uk': 'scope-repair-66'}


CONFIGS = {0: (2, 20), 1: (2, 50), 2: (3, 20), 3: (3, 50)}
REGULARIZATION = tuple(CONFIGS)
from training.three_expert_fusion_v9 import load_split as parent_split
from training.conservative_fusion_v9 import full_coverage_gap

def semantic_vector(vector):
    if vector is None: return None
    width = sum(map(len, ONTOLOGY.values()))
    if len(vector) != 2 + 7*width: raise ValueError('Parent feature dimension differs')
    return vector[2:2+2*width] + vector[2+3*width:2+5*width] + vector[2+6*width:]

def load_split(root, language, split):
    rows, vectors, truth, hashes = parent_split(root, language, split)
    return rows, [semantic_vector(v) for v in vectors], truth, hashes

import math
import struct


def export_model(model):
    import numpy as np
    import sklearn
    if model.classes_.tolist() != [0, 1]:
        raise ValueError('Expected binary concern classes')
    prior = float(model.init_.predict_proba(np.zeros((1, model.n_features_in_)))[0, 1])
    if not 0 < prior < 1:
        raise ValueError('Invalid initial class prior')
    trees = []
    for estimator in model.estimators_[:, 0]:
        tree = estimator.tree_
        trees.append(dict(feature=tree.feature.tolist(), threshold=tree.threshold.tolist(),
            left=tree.children_left.tolist(), right=tree.children_right.tolist(),
            value=tree.value[:, 0, 0].tolist()))
    return dict(format='roguard-boosted-score-v1',dimension=int(model.n_features_in_),
        intercept=math.log(prior/(1-prior)),learning_rate=float(model.learning_rate),
        trees=trees,sklearn_version=sklearn.__version__)


def fit(x, y, kinds, regularization):
    from sklearn.ensemble import GradientBoostingClassifier
    groups=list(zip(kinds,y));frequencies=Counter(groups)
    if len(frequencies)!=4:raise ValueError('Fit split lacks a class/style stratum')
    weights=[len(y)/(4*frequencies[g]) for g in groups]
    depth,leaf=CONFIGS[regularization]
    model=GradientBoostingClassifier(loss='log_loss',n_estimators=100,learning_rate=.05,
        max_depth=depth,min_samples_leaf=leaf,subsample=1.,random_state=978,n_iter_no_change=None)
    model.fit(x,y,sample_weight=weights)
    exported=export_model(model)
    # Check the portable arithmetic before the model leaves this process.
    reference=model.predict_proba(x)[:,1]
    if max(abs(float(a)-probability(exported,b)) for a,b in zip(reference,x))>1e-12:
        raise ValueError('Portable boosted tree probabilities differ')
    return exported


def probability(model, vector):
    if vector is None:return None
    if len(vector)!=model['dimension']:raise ValueError('Boosted feature dimension differs')
    values=[struct.unpack('f',struct.pack('f',float(v)))[0] for v in vector]
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite boosted feature')
    logit=model['intercept']
    for tree in model['trees']:
        node=0
        while tree['left'][node]!=-1:
            node=tree['left'][node] if values[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
        logit+=model['learning_rate']*tree['value'][node]
    return 1/(1+math.exp(-max(-700,min(700,logit))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'audit'))
    parser.add_argument('--language', choices=('uk',), required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization'); output = root/'semantic-boosted-81'/args.language
    protocol = root/'semantic-boosted-81/protocol.json'
    parent_paths = [root/('native-full-22-'+args.language)/'review.json.gz', root/FIELD_STUDIES[args.language]/args.language/'selection.json', root/'interpreter-full-21/selection.json']
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
            limit = full_coverage_gap(y, scores); counts = metrics(rows, y, scores, limit)
            # Unchanged gate eligibility precedes false-review ranking.
            rank = (not all(v['passes'] for v in counts.values()), max(v['false_review_rate'] for v in counts.values()),
                    sum(v['false_review_rate'] for v in counts.values()), CONFIGS[regularization][0], -CONFIGS[regularization][1])
            trials.append(dict(regularization=regularization, threshold=limit, metrics=counts, rank=rank, scores=scores))
        selected = min(trials, key=lambda v: v['rank'])
        valid = [i for i in range(len(rows)) if x[i] is not None]
        fitted = fit([x[i] for i in valid], [y[i] for i in valid], [kinds[i] for i in valid], selected['regularization'])
        refitted = [probability(fitted, v) for v in x]
        selected['threshold'] = min(selected['threshold'], full_coverage_gap(y+y, selected['scores']+refitted))
        selected['metrics'] = metrics(rows, y, selected['scores'], selected['threshold'])
        selected['refit_metrics'] = metrics(rows, y, refitted, selected['threshold'])
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
                      counts_pass=all(v['passes'] for v in counts.values()) and all(v['passes'] for v in record['selected']['metrics'].values()) and all(v['passes'] for v in record['selected']['refit_metrics'].values()),
                      stable_release_ready=False, reserved_probes_used=False,
                      requirements_remaining=['Installed composite runtime parity', 'Frozen reserved actual-runtime probes', 'Product integration and release checks'])
        with (output/'cal-predictions.jsonl').open('x') as stream:
            stream.write(''.join(json.dumps(dict(case_id=r['case_id'], gold=t, score=s,
                predicted=None if s is None else int(s >= record['selected']['threshold'])))+'\n' for r,t,s in zip(rows,y,scores)))
        (output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
