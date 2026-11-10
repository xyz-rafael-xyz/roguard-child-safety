"""Strong comparator: reselect v9's existing score mode on the same development."""
import argparse
import json
from pathlib import Path

from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.native_fields_v9 import SIZES, policy_mass
from training.separation_frontier_v9 import frontier
from training.prepare_instruction_sft_v8 import target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('features','feature_study','development_study','candidate','study'):
        p.add_argument('--'+key.replace('_','-'),dest=key,type=Path,required=True)
    a=p.parse_args()
    if a.study.exists():raise ValueError('Refusing to overwrite a baseline selection')
    extraction=json.loads((a.feature_study/'report.json').read_text())
    added=json.loads((a.development_study/'report.json').read_text())
    if not extraction['complete'] or not added['complete']:raise ValueError('Incomplete development representations')
    if file_sha(a.development_study/'inputs.jsonl')!=added['inputs_sha256']:raise ValueError('Additional development differs')
    new=read_jsonl(a.development_study/'inputs.jsonl')
    if any(r['partition']!='dev' for r in new):raise ValueError('Only development is allowed')
    for lang,record in extraction['languages'].items():
        for suffix,sha in record['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix))!=sha:raise ValueError('Frozen v9 representation differs')
    a.study.mkdir(parents=True)
    protocol={'format':'roguard-frozen-score-baseline-1','source_sha256':file_sha(Path(__file__)),
        'feature_report_sha256':file_sha(a.feature_study/'report.json'),
        'trajectory_development_report_sha256':file_sha(a.development_study/'report.json'),
        'purpose':'Determine whether selecting an existing v9 head/mode is sufficient, before attributing gains to new training.',
        'model_weights_changed':False,'selection':'All original and eligible trajectory development; errors then largest score margin.',
        'calibration_or_test_loaded':False,'independent_test':False,'stable_release_ready':False}
    (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    report={'complete':False,'languages':{},'stable_release_ready':False,'weights_changed':False}
    for lang in ('ro','uk'):
        metadata=read_jsonl(a.features/(lang+'-rows.jsonl'))
        all_z=np.load(a.features/(lang+'-logits.npy'),mmap_mode='r')
        indices=[i for i,r in enumerate(metadata) if r['partition']=='dev']
        rows=[metadata[i] for i in indices];z=[all_z[i] for i in indices]
        for r in new:
            if r['row']['language']!=lang:continue
            rows.append({'case_id':r['row']['case_id'],'language':lang,'group':r['group'],'gold':target(r['row'])})
            z.append(r['reference_logits'])
        z=np.array(z,dtype=np.float64)
        def softmax(v):
            exp=np.exp(v-v.max(axis=1,keepdims=True));return exp/exp.sum(axis=1,keepdims=True)
        spec=json.loads((a.feature_study/'v9-binding'/(lang+'-review.json')).read_text())
        if file_sha(a.candidate/'actual'/(lang+'-selected.safetensors'))!=spec['adapter_sha256']:
            raise ValueError('Frozen baseline adapter differs')
        factors=[softmax(v) for v in np.split(z[:,:sum(SIZES)],np.cumsum(SIZES)[:-1],axis=1)]
        modes={'binary':softmax(z[:,-2:])[:,1],'policy':policy_mass(factors),'coupled_policy':policy_mass(factors,spec['couplings'])}
        options=[];best=None
        for mode,scores in modes.items():
            population=[{**r,'score':float(score)} for r,score in zip(rows,scores)]
            boundary=frontier(population)
            for row in population:row['predicted']=int(row['score']>=boundary['threshold'])
            errors=sum(r['predicted']!=r['gold'] for r in population)
            rank=[errors,-boundary['score_margin']]
            record={'mode':mode,'threshold':boundary['threshold'],'binary_errors':errors,
                    'score_margin':boundary['score_margin'],'selection_rank':rank}
            options.append(record)
            if best is None or rank<best['selection_rank']:
                best=record
                write_jsonl(a.study/(lang+'-development-predictions.jsonl'),population)
        report['languages'][lang]={'development_rows':len(rows),'modes':options,'selected':best}
        spec.update(mode=best['mode'],threshold=best['threshold'])
        (a.study/(lang+'-review.json')).write_text(json.dumps(spec,indent=2)+'\n')
    report['complete']=True
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
