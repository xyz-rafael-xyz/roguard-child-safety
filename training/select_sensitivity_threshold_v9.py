"""Inspect the lowest cutoff permitted by each development false-review budget.

This is a research diagnostic, not runtime calibration or a release decision.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
from training.check_review_candidate_v8 import check_counts
from training.compare_review_candidates_v9 import read


def evaluate(candidate, language):
    data = {}
    for split in ('dev','cal'):
        # Validate complete coverage and agreement with the saved gold labels.
        read(candidate, split, language)
        source = {r['case_id']:r for r in map(json.loads,gzip.decompress((candidate/(split+'.jsonl.gz')).read_bytes()).decode().splitlines())}
        data[split] = [(p,'conversational' if source[p['case_id']].get('source')=='v7_training_frame_projection' else 'formal')
            for p in map(json.loads,(candidate/(split+'-predictions.jsonl')).read_text().splitlines()) if p['language']==language]
        if any(type(p['score']) not in (int,float) or not math.isfinite(p['score']) or not 0<=p['score']<=1 for p,k in data[split]):
            raise ValueError('Invalid probability')
    limits = []
    for kind in ('formal','conversational'):
        negatives = sorted((p['score'] for p,k in data['dev'] if k==kind and p['gold']==0),reverse=True)
        if not negatives:
            raise ValueError('Missing development negatives')
        limits.append(math.nextafter(negatives[len(negatives)*5//100],math.inf))
    threshold = max(limits)
    result = dict(language=language,threshold=threshold,stable_release_ready=False,selection_partition='development',metrics={})
    for split, ps in data.items():
        metrics = {}
        for kind in ('formal','conversational'):
            rows = [p for p,k in ps if k==kind]
            metrics[language+'_'+kind] = dict(positive=sum(p['gold']==1 for p in rows),negative=sum(p['gold']==0 for p in rows),
                missed=sum(p['gold']==1 and p['score']<threshold for p in rows),
                false_reviews=sum(p['gold']==0 and p['score']>=threshold for p in rows))
        result['metrics'][split] = metrics
    result['failures'] = [s+'/'+f for s,m in result['metrics'].items() for f in check_counts(m,(language+'_formal',language+'_conversational'))]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--language',choices=('ro','uk'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(evaluate(a.candidate,a.language),indent=2)+'\n')


if __name__=='__main__':
    main()
