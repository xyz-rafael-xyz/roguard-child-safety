"""Train a quotation-abstracted scope gate independently of concern content.

The scope target asks whether the policy would review a concern if present.
Multiplication with a frozen concern-sensitive encoder produces the candidate
score. Selection uses development only; calibration is a known regression set.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from roguard.evidence import disposition
from training.quote_scope_v9 import scope_text
from training.natural_contrasts_v9 import generate as natural
from training.operator_contrasts_v9 import generate as operators
from training.review_data_v8 import input_key, isolate_partitions
from training.fit_sparse_review_v9 import read, metrics, select_threshold
from training.compare_review_candidates_v9 import read as validate_predictions
from training.check_review_candidate_v8 import check_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--language',choices=('ro','uk'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args();a.output.mkdir(exist_ok=False);threadpool_limits(2)
    history = read(a.baseline/'training-history.jsonl.gz')
    rows = {'train':[r for r in history+natural()+operators() if r['language']==a.language]}
    for s in ('dev','cal'):
        validate_predictions(a.baseline,s,a.language)
        rows[s] = [r for r in read(a.baseline/(s+'.jsonl.gz')) if r['language']==a.language]
    rows, removed = isolate_partitions(rows,history)
    if removed['dev'] or removed['cal']:
        raise ValueError('Evaluation population changed')
    for s,rs in rows.items():
        (a.output/(s+'.jsonl.gz')).write_bytes(gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs).encode(),mtime=0))
    (a.output/'overlap-exclusions.json').write_text(json.dumps(removed,indent=2)+'\n')
    y = {s:np.array([int(disposition({k:{'label':v} for k,v in r['attributes'].items()})[0]!='CONTINUE') for r in rs]) for s,rs in rows.items()}
    scope_y = np.array([int(disposition({k:{'label':'1' if k=='concern' else v} for k,v in r['attributes'].items()})[0]!='CONTINUE') for r in rows['train']])
    vectorizers=[];blocks={s:[] for s in rows}
    for focus,analyzer,ngrams,size in ((True,'char',(2,5),16000),(True,'word',(1,3),12000),(False,'word',(1,2),6000)):
        vec=TfidfVectorizer(analyzer=analyzer,ngram_range=ngrams,min_df=2,max_features=size,sublinear_tf=True)
        blocks['train'].append(vec.fit_transform([scope_text(r,focus) for r in rows['train']]))
        for s in ('dev','cal'):blocks[s].append(vec.transform([scope_text(r,focus) for r in rows[s]]))
        vectorizers.append(dict(focus=focus,analyzer=analyzer,ngram_range=ngrams,vocabulary=vec.vocabulary_,idf=vec.idf_.tolist()))
    x={s:hstack(v).tocsr() for s,v in blocks.items()}
    baseline={}
    for s in ('dev','cal'):
        lookup={p['case_id']:p['score'] for p in map(json.loads,(a.baseline/(s+'-predictions.jsonl')).read_text().splitlines())}
        baseline[s]=np.array([lookup[r['case_id']] for r in rows[s]])
    record=dict(language=a.language,synthetic_only=True,stable_release_ready=False,reserved_probes_used=False,
        baseline=str(a.baseline),baseline_sha256=hashlib.sha256((a.baseline/'review.json.gz').read_bytes()).hexdigest(),
        selection_partition='development',training_rows=len(rows['train']),candidates=[])
    best=None
    for c in (.1,1.,10.):
        head=LogisticRegression(C=c,max_iter=1000,class_weight='balanced',random_state=910).fit(x['train'],scope_y)
        scope=head.predict_proba(x['dev'])[:,1]
        for power in (1,2,4):
            scores=baseline['dev']*scope**power
            threshold=select_threshold(rows['dev'],y['dev'],scores)
            m=metrics(rows['dev'],y['dev'],scores,threshold,a.language)
            rates=[v['false_reviews']/v['negative'] for v in m.values()]
            rank=(max(rates),sum(rates),power,c)
            entry=dict(C=c,power=power,threshold=threshold,development=m)
            record['candidates'].append(entry);print(json.dumps(entry),flush=True)
            if best is None or rank<best:
                best=rank;chosen=head;selected=entry;chosen_dev=scores
    blob=gzip.compress(json.dumps(dict(format='roguard-scope-gate-experiment-v1',vectorizers=vectorizers,
        weights=chosen.coef_[0].tolist(),intercept=float(chosen.intercept_[0]),selected=selected,
        baseline_sha256=record['baseline_sha256']),separators=(',',':'),default=int).encode(),mtime=0)
    (a.output/'scope-gate.json.gz').write_bytes(blob)
    cal=baseline['cal']*chosen.predict_proba(x['cal'])[:,1]**selected['power']
    record.update(selected=selected,artifact_sha256=hashlib.sha256(blob).hexdigest(),calibration=metrics(rows['cal'],y['cal'],cal,selected['threshold'],a.language))
    for s,scores in (('dev',chosen_dev),('cal',cal)):
        predictions=[dict(case_id=r['case_id'],language=a.language,gold=int(t),score=float(v),predicted=int(v>=selected['threshold'])) for r,t,v in zip(rows[s],y[s],scores)]
        (a.output/(s+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    groups=(a.language+'_formal',a.language+'_conversational')
    record['failures']=['dev/'+v for v in check_counts(selected['development'],groups)]+['cal/'+v for v in check_counts(record['calibration'],groups)]
    (a.output/'development.json').write_text(json.dumps(record,indent=2)+'\n');print('FINISHED',json.dumps(record['calibration']),flush=True)


if __name__=='__main__':main()
