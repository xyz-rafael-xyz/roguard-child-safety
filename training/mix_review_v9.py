"""Development-selected score mixtures; calibration is only a regression audit.

Mixtures do not consume fresh probes or change the acceptance thresholds.
The output describes a research ensemble, not a deployable review adapter.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from training.compare_review_candidates_v9 import read
from training.check_review_candidate_v8 import check_counts


def logit(score):
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid probability')
    score = min(1-1e-7, max(1e-7, score))
    return math.log(score/(1-score))


def evaluate(baseline, candidate, language):
    data = {}
    hashes = {}
    for split in ('dev', 'cal'):
        old, old_hash = read(baseline, split, language)
        new, new_hash = read(candidate, split, language)
        if not old or set(old) != set(new) or any((old[k][0],old[k][2]) != (new[k][0],new[k][2]) for k in old):
            raise ValueError('Evaluation inputs or labels differ')
        hashes[split] = {'baseline': old_hash, 'candidate': new_hash}
        sources = []
        for folder in (baseline, candidate):
            rows = {r['case_id']: r for r in map(json.loads, gzip.decompress((Path(folder)/(split+'.jsonl.gz')).read_bytes()).decode().splitlines())}
            from training.review_data_v8 import input_key
            sources.append({input_key(rows[p['case_id']]): logit(p['score']) for p in map(json.loads,(Path(folder)/(split+'-predictions.jsonl')).read_text().splitlines()) if p['language']==language})
        data[split] = [(k, old[k][0], old[k][2], sources[0][k], sources[1][k]) for k in old]
    groups = (language+'_formal', language+'_conversational')
    def metrics(rows, weight, threshold):
        result = {g: dict(positive=0,negative=0,missed=0,false_reviews=0) for g in groups}
        for _, gold, group, a, b in rows:
            prediction = (1-weight)*a+weight*b >= threshold
            m = result[group]
            m['positive' if gold else 'negative'] += 1
            m['missed'] += int(gold==1 and not prediction)
            m['false_reviews'] += int(gold==0 and prediction)
        return result
    experiments = []
    for weight in (0, .25, .5, .75, 1):
        cutoffs = []
        for group in groups:
            positive = sorted((1-weight)*a+weight*b for _,gold,g,a,b in data['dev'] if gold==1 and g==group)
            if not positive: raise ValueError('Missing positive group')
            cutoffs.append(positive[len(positive)//100])
        threshold = min(cutoffs)-1e-7
        m = metrics(data['dev'], weight, threshold)
        if any(v['negative']==0 for v in m.values()): raise ValueError('Missing benign group')
        rates = [v['false_reviews']/v['negative'] for v in m.values()]
        experiments.append({'candidate_weight':weight,'logit_threshold':threshold,'development':m,'rank':[max(rates),sum(rates),weight]})
    selected = min(experiments, key=lambda x:x['rank'])
    calibration = metrics(data['cal'], selected['candidate_weight'], selected['logit_threshold'])
    failures = ['dev/'+v for v in check_counts(selected['development'],groups)] + ['cal/'+v for v in check_counts(calibration,groups)]
    return {'synthetic_regression_only':True,'stable_release_ready':False,'language':language,'baseline':str(baseline),'candidate':str(candidate),'input_sha256':hashes,
            'selection_partition':'development','candidates':experiments,'selected':selected,'calibration':calibration,'failures':failures,'prerequisites_pass':not failures}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--language',choices=('ro','uk'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=evaluate(a.baseline,a.candidate,a.language)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'language':a.language,'selected_weight':result['selected']['candidate_weight'],'failures':result['failures']}))


if __name__=='__main__':main()
