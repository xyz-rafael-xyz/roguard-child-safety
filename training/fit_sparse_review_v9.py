"""Train-only lexical scope heads, selected on fixed development populations.

Research only. Known calibration is a regression audit, not fresh validation.
No reserved probes are read. Export uses JSON rather than executable pickle.
"""
import argparse
import gzip
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from roguard.evidence import ONTOLOGY, disposition
from training.operator_contrasts_v9 import generate
from training.policy_probability_v9 import review_mass
from training.review_data_v8 import input_key
from training.check_review_candidate_v8 import check_counts


def read(path):
    return [json.loads(s) for s in gzip.decompress(path.read_bytes()).decode().splitlines()]


def texts(row, focus):
    turns = row['turns'][:row['focus']+1]
    text = turns[-1]['text'] if focus else '\n'.join(t['text'] for t in turns[:-1])
    return unicodedata.normalize('NFC', text)


def metrics(rows, y, scores, threshold, language):
    result = {}
    for kind in ('formal', 'conversational'):
        ix = np.array([(r.get('source') == 'v7_training_frame_projection') == (kind == 'conversational') for r in rows])
        truth, prediction = y[ix], scores[ix] >= threshold
        result[language+'_'+kind] = dict(positive=int(sum(truth)), negative=int(sum(1-truth)),
            missed=int(sum((truth == 1) & ~prediction)), false_reviews=int(sum((truth == 0) & prediction)))
    return result


def select_threshold(rows, y, scores):
    limits = []
    for native in (False, True):
        positive = sorted(float(p) for r, target, p in zip(rows, y, scores)
                          if target and (r.get('source') == 'v7_training_frame_projection') == native)
        if not positive:
            raise ValueError('Missing positive development group')
        limits.append(positive[len(positive)//100])
    return min(limits)-1e-8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--language', choices=('ro', 'uk'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    threadpool_limits(2)
    history = read(args.baseline/'training-history.jsonl.gz')
    train = history + generate()
    train = list({input_key(r): r for r in train if r['language'] == args.language}.values())
    rows = {'train': train}
    for split in ('dev', 'cal'):
        rows[split] = [r for r in read(args.baseline/(split+'.jsonl.gz')) if r['language'] == args.language]
    keys = {s: {input_key(r) for r in rs} for s, rs in rows.items()}
    if any(keys[a] & keys[b] for a, b in (('train','dev'),('train','cal'),('dev','cal'))):
        raise ValueError('Overlapping partitions')
    for s, rs in rows.items():
        raw = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rs).encode()
        (args.output/(s+'.jsonl.gz')).write_bytes(gzip.compress(raw, mtime=0))
    y = {s: np.array([int(disposition({k:{'label':v} for k,v in r['attributes'].items()})[0]!='CONTINUE') for r in rs]) for s,rs in rows.items()}
    blocks = {s: [] for s in rows}
    vectorizers = []
    # Explicitly separate the current message from preceding context.
    for focus, analyzer, ngrams, size in ((True,'char',(2,5),16000),(True,'word',(1,3),12000),(False,'word',(1,2),6000)):
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, min_df=2, max_features=size, sublinear_tf=True)
        blocks['train'].append(v.fit_transform([texts(r,focus) for r in train]))
        for s in ('dev','cal'):
            blocks[s].append(v.transform([texts(r,focus) for r in rows[s]]))
        vectorizers.append(dict(focus=focus, analyzer=analyzer, ngram_range=ngrams,
            vocabulary=v.vocabulary_, idf=v.idf_.tolist()))
    x = {s: hstack(b).tocsr() for s,b in blocks.items()}
    record = dict(language=args.language, synthetic_only=True, reserved_probes_used=False,
        stable_release_ready=False, selection_partition='development', candidates=[], training_rows=len(train))
    best = None
    for c in (1., 10.):
        heads = {}
        for task in ('review', *ONTOLOGY):
            targets = y['train'] if task=='review' else [r['attributes'][task] for r in train]
            model = LogisticRegression(C=c, max_iter=1000, class_weight='balanced', random_state=905)
            model.fit(x['train'], targets)
            heads[task] = model
        for mode in ('direct', 'policy'):
            def score(split):
                if mode=='direct': return heads['review'].predict_proba(x[split])[:,1]
                p = {}
                for task in ONTOLOGY:
                    values = heads[task].predict_proba(x[split])
                    p[task] = {str(k): values[:,i] for i,k in enumerate(heads[task].classes_)}
                return np.clip(review_mass(p), 0, 1)
            scores = score('dev')
            threshold = select_threshold(rows['dev'],y['dev'],scores)
            m = metrics(rows['dev'],y['dev'],scores,threshold,args.language)
            rates = [v['false_reviews']/v['negative'] for v in m.values()]
            rank = (max(rates), sum(rates), c, mode)
            entry = dict(C=c,mode=mode,threshold=threshold,development=m)
            record['candidates'].append(entry)
            print(json.dumps(entry),flush=True)
            if best is None or rank < best:
                best = rank
                selected = entry
                chosen_scores = {s:score(s) for s in ('dev','cal')}
                chosen = {t:dict(classes=[str(k) for k in h.classes_],weights=h.coef_.tolist(),intercept=h.intercept_.tolist()) for t,h in heads.items()}
    artifact = dict(format='roguard-sparse-review-experiment-v1',heads=chosen,vectorizers=vectorizers,selected=selected)
    blob = gzip.compress(json.dumps(artifact,separators=(',',':'),default=int).encode(),mtime=0)
    (args.output/'sparse.json.gz').write_bytes(blob)
    record.update(selected=selected,artifact_sha256=hashlib.sha256(blob).hexdigest())
    for split, scores in chosen_scores.items():
        predictions = [dict(case_id=r['case_id'],language=r['language'],gold=int(t),score=float(p),predicted=int(p>=selected['threshold'])) for r,t,p in zip(rows[split],y[split],scores)]
        (args.output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    record['calibration'] = metrics(rows['cal'],y['cal'],chosen_scores['cal'],selected['threshold'],args.language)
    groups = (args.language+'_formal',args.language+'_conversational')
    record['failures'] = ['dev/'+s for s in check_counts(selected['development'],groups)]+['cal/'+s for s in check_counts(record['calibration'],groups)]
    (args.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record['calibration']),flush=True)


if __name__=='__main__':
    main()
