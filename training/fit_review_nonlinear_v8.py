"""Test whether nonlinear combinations of frozen features reduce review errors."""
import argparse
import hashlib
import gzip
import json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from roguard.context import render
from training.fit_evidence_v8 import semantic_features
from training.fit_review_v8 import labels, group_metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('research/v8-stabilization/review-01'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.input
    output=args.output;output.mkdir(exist_ok=False)
    rows={s:[json.loads(l) for l in gzip.decompress((root/(s+'.jsonl.gz')).read_bytes()).decode().splitlines()] for s in ('train','dev','cal')}
    matrix={};target={}
    for split,rs in rows.items():
        context=semantic_features([render(r['turns'][:r['focus']+1]) for r in rs],None,root/(split+'-context-features.npz'))
        focus=semantic_features([r['turns'][r['focus']]['text'] for r in rs],None,root/(split+'-focus-features.npz'))
        matrix[split]=np.concatenate([context,focus],axis=1);target[split]=labels(rs)
    record={'selection_partition':'development','synthetic_only':True,'input_sha256':{s:hashlib.sha256((root/(s+'.jsonl.gz')).read_bytes()).hexdigest() for s in rows},'candidates':[]};best=None
    for width in (64,128):
        head=MLPClassifier(hidden_layer_sizes=(width,),alpha=.1,max_iter=100,early_stopping=True,random_state=862).fit(matrix['train'],target['train'])
        p=head.predict_proba(matrix['dev'])[:,1];thresholds={}
        for lang in ('ro','uk'):
            limits=[]
            for native in (False,True):
                scores=sorted(float(p[i]) for i,r in enumerate(rows['dev']) if r['language']==lang and (r.get('source')=='v7_training_frame_projection')==native and target['dev'][i])
                limits.append(scores[int(.01*len(scores))])
            thresholds[lang]=min(limits)
        metrics=group_metrics(rows['dev'],target['dev'],p,thresholds)
        rank=(max(v['false_review_rate'] for v in metrics.values()),sum(v['false_review_rate'] for v in metrics.values()))
        entry={'width':width,'iterations':head.n_iter_,'thresholds':thresholds,'development':metrics}
        record['candidates'].append(entry);print(json.dumps(entry),flush=True)
        if best is None or rank<best:best=rank;chosen=head;chosen_thresholds=thresholds;record['selected_width']=width
    record['calibration_audit']=group_metrics(rows['cal'],target['cal'],chosen.predict_proba(matrix['cal'])[:,1],chosen_thresholds)
    (output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    (output/'review-head.json').write_text(json.dumps({'weights':[v.tolist() for v in chosen.coefs_],'intercepts':[v.tolist() for v in chosen.intercepts_],'activation':'relu','thresholds':chosen_thresholds})+'\n')
    print('FINISHED',json.dumps(record['calibration_audit']),flush=True)
if __name__=='__main__':main()
