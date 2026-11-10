"""Pair crossed-scope readout and frozen v9 on each original and generated DEV population."""
import argparse
import json
from pathlib import Path

from roguard.gemma4_review import _score
from roguard.trajectory_data import file_sha,read_jsonl,write_jsonl
from training.shared_tail_training_v9 import make_head


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','feature_study','selected','selected_study','original','study','private'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    a=p.parse_args()
    if a.study.exists() or a.private.exists():raise ValueError('Refusing to overwrite assessment')
    extraction=json.loads((a.feature_study/'report.json').read_text());selection=json.loads((a.selected_study/'report.json').read_text())
    if not extraction['complete'] or not selection['complete']:raise ValueError('Incomplete source')
    for lang,record in extraction['languages'].items():
        for suffix,expected in record['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix))!=expected:raise ValueError('Feature binding differs')
    import numpy as np
    import mlx.core as mx
    from safetensors.numpy import load_file
    groups={};records=[];maximum=0.;batch_disagreements=0
    for lang in ('ro','uk'):
        rows=read_jsonl(a.features/(lang+'-rows.jsonl'));dev=[i for i,row in enumerate(rows) if row['partition']=='dev']
        spec=json.loads((a.selected_study/(lang+'-review.json')).read_text())
        original_spec=json.loads((a.feature_study/'v9-binding'/(lang+'-review.json')).read_text())
        if original_spec['adapter_sha256']!=file_sha(a.original/'actual'/(lang+'-selected.safetensors')):
            raise ValueError('Original adapter binding differs')
        old=load_file(a.original/'actual'/(lang+'-selected.safetensors'));new=load_file(a.selected/(lang+'-selected.safetensors'))
        if spec['adapter_sha256']!=file_sha(a.selected/(lang+'-selected.safetensors')) or spec['adapter_sha256']!=selection['languages'][lang]['selected']['adapter_sha256']:
            raise ValueError('Selected adapter binding differs')
        if set(old)!=set(new) or any(not np.array_equal(old[key],new[key]) for key in old if not key.startswith('heads.'+lang+'.')):
            raise ValueError('Frozen encoder differs')
        head=make_head({key:new['heads.'+lang+'.'+value] for key,value in
                       {'0.weight':'first.weight','0.bias':'first.bias','3.weight':'last.weight','3.bias':'last.bias'}.items()});head.eval()
        x=np.load(a.features/(lang+'-features.npy'),allow_pickle=False)
        teacher=np.load(a.features/(lang+'-logits.npy'),allow_pickle=False)
        saved=np.load(a.selected/(lang+'-selected-dev-logits.npy'),allow_pickle=False)
        if x.shape!=(len(rows),2560) or teacher.shape!=(len(rows),19) or saved.shape!=(len(dev),19):raise ValueError('Feature/evaluation shape differs')
        for j,i in enumerate(dev):
            z=head(mx.array(x[i:i+1]));mx.eval(z);z=np.asarray(z)[0]
            if not np.isfinite(z).all():raise ValueError('Nonfinite selected logits')
            error=float(np.max(np.abs(z-saved[j])));maximum=max(maximum,error)
            if error>1e-3:raise ValueError('Selected single-input reload parity failed')
            new_score=_score(z,spec);old_score=_score(teacher[i],original_spec)
            predicted=int(new_score>=spec['threshold']);baseline=int(old_score>=original_spec['threshold'])
            batch_disagreements+=predicted!=int(_score(saved[j],spec)>=spec['threshold'])
            row=rows[i];origin='crossed' if row['case_id'].startswith('composition61:') else 'generated' if row.get('family','').startswith('composition56:') else 'historical'
            g=groups.setdefault(origin+':'+lang,{'rows':0,'candidate_errors':0,'v9_errors':0,'fixed':0,'introduced':0})
            g['rows']+=1;g['candidate_errors']+=predicted!=row['gold'];g['v9_errors']+=baseline!=row['gold']
            g['fixed']+=predicted==row['gold'] and baseline!=row['gold'];g['introduced']+=predicted!=row['gold'] and baseline==row['gold']
            records.append({'case_id':row['case_id'],'language':lang,'origin':origin,'gold':row['gold'],
                            'candidate':predicted,'v9':baseline,'candidate_score':new_score,'v9_score':old_score,'token_sha256':row['token_sha256']})
    if {key:value['rows'] for key,value in groups.items()}!={'historical:ro':2954,'historical:uk':3387,'generated:ro':288,'generated:uk':288,'crossed:ro':960,'crossed:uk':960}:
        raise ValueError('Required development populations differ')
    if len({(r['language'],r['case_id']) for r in records})!=8837:raise ValueError('Incomplete or duplicate development outputs')
    a.study.mkdir(parents=True);a.private.mkdir(parents=True);write_jsonl(a.private/'predictions.jsonl',records)
    report={'complete':True,'source_sha256':file_sha(__file__),'feature_report_sha256':file_sha(a.feature_study/'report.json'),
            'selection_report_sha256':file_sha(a.selected_study/'report.json'),'groups':groups,
            'predictions_sha256':file_sha(a.private/'predictions.jsonl'),'maximum_batch_single_logit_error':maximum,
            'batch_single_decision_disagreements':batch_disagreements,
            'decision':'REJECT' if any(g['candidate_errors'] or g['introduced'] for g in groups.values()) or batch_disagreements else 'DEVELOPMENT_PASS_ONLY',
            'held_out_test_opened':False,'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
