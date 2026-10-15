"""Audit predeclared input-length calibration without fitting on calibration labels."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from training.fit_review_v8 import group_metrics, labels


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    spec=json.loads(gzip.decompress((args.candidate/'review.json.gz').read_bytes()))
    rows={}; scores={}
    for split in ('dev','cal'):
        rows[split]=[json.loads(l) for l in gzip.decompress((args.candidate/(split+'.jsonl.gz')).read_bytes()).decode().splitlines()]
        predictions=[json.loads(l) for l in (args.scores/split/'predictions.jsonl').read_text().splitlines()]
        assert len(rows[split])==len(predictions)
        assert all(r['case_id']==p['case_id'] and r['language']==p['language'] for r,p in zip(rows[split],predictions))
        scores[split]=np.array([p['score'] for p in predictions])
    def key(row):
        return row['language']+('_short' if len(row['turns'][row['focus']]['text'].split())<=12 else '_long')
    truth=labels(rows['dev']);thresholds={};counts={}
    for lang in ('ro','uk'):
        for length in ('short','long'):
            k=lang+'_'+length
            values=sorted(float(s) for r,y,s in zip(rows['dev'],truth,scores['dev']) if key(r)==k and y)
            counts[k]=len(values)
            thresholds[k]=values[int(.01*len(values))]*(1-1e-4) if len(values)>=100 else spec['thresholds'][lang]
    result={'synthetic_only':True,'thresholds':thresholds,'development_positive_counts':counts,'selection_partition':'development','fresh_probe_used':False,'partitions':{}}
    for split in ('dev','cal'):
        truth=labels(rows[split]);pred=np.array([int(s>=thresholds[key(r)]) for r,s in zip(rows[split],scores[split])])
        result['partitions'][split]={'language_only':group_metrics(rows[split],truth,scores[split],spec['thresholds']),
                                   'length_conditioned':group_metrics(rows[split],truth,pred,{'ro':.5,'uk':.5})}
    result['passes_all_groups']=all(v['recall']>=.99 and v['false_review_rate']<=.05 for p in result['partitions'].values() for v in p['length_conditioned'].values())
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
