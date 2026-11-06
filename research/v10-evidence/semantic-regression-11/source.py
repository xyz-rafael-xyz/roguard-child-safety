"""Compare frozen selected semantic heads on exposed regressions, without fitting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.evidence import ONTOLOGY
from roguard.gemma4_review import _score
from roguard.trajectory_data import file_sha,input_identity,prefix,read_jsonl,write_jsonl
from roguard.trajectory_evaluation import checkpoint_id,evaluate,paired_compare
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import FieldEngine,prepare_memory
from training.native_fields_v9 import SIZES,FIELDS
from training.shared_tail_training_v9 import make_head


class PersistentPrefixEngine(FieldEngine):
    cache_instructions=True


def world_from_logits(values):
    world={};offset=0
    for field,size in zip(FIELDS,SIZES):
        section=values[offset:offset+size]
        world[field]=ONTOLOGY[field][max(range(size),key=lambda i:section[i])];offset+=size
    return world


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base','candidate','selected','selected_study','selector','study','private'):
        p.add_argument('--'+key.replace('_','-'),dest=key,type=Path,required=True)
    a=p.parse_args()
    if a.study.exists() or a.private.exists():raise SystemExit('Refusing to overwrite qualification')
    selection=json.loads((a.selected_study/'report.json').read_text())
    if not selection.get('complete'):raise ValueError('Model selection is not complete')
    a.study.mkdir(parents=True);a.private.mkdir(parents=True)
    source=Path('research/v10-evidence')
    paths={'broad':source/'broad-v9-regression/cases.jsonl',
           **{name:source/'data'/(name+'.jsonl') for name in ('dev','challenge','v9-regression')}}
    cases={name:read_jsonl(path) for name,path in paths.items()}
    selector=TrajectoryModel(a.selector);frozen=LocalFrozenV9(a.base,a.candidate,a.study/'v9-binding')
    original_scores={}
    for lang in ('ro','uk'):
        for r in read_jsonl(Path('research/v9-stabilization/reported-tail-prior-197/actual')/(lang+'-prior-predictions.jsonl')):
            original_scores[(lang,r['group'],r['case_id'])]=r
    protocol={'format':'roguard-v10-semantic-regression-1','selection_report_sha256':file_sha(a.selected_study/'report.json'),
              'source_sha256':file_sha(Path(__file__)),'inputs':{k:file_sha(v) for k,v in paths.items()},
              'selected_adapters':{lang:file_sha(a.selected/(lang+'-selected.safetensors')) for lang in ('ro','uk')},
              'selector_sha256':selector.identity,'existing_exposed_regressions_only':True,
              'fitting_performed':False,'threshold_selection_performed':False,'stable_release_ready':False,
              'interpretation_note':'Singleton argmax labels are uncalibrated diagnostic predictions, not coverage-qualified sets.'}
    (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    import mlx.nn as nn
    from safetensors.numpy import load_file
    output={name:{'baseline':[],'candidate':[]} for name in cases};done=0;started=time.monotonic();max_error=0
    for lang in ('ro','uk'):
        old=load_file(a.candidate/'actual'/(lang+'-selected.safetensors'))
        new=load_file(a.selected/(lang+'-selected.safetensors'))
        if set(old)!=set(new) or any(not np.array_equal(old[k],new[k]) for k in old if not k.startswith('heads.'+lang+'.')):
            raise ValueError('Candidate changed the frozen encoder or another language head')
        selected_spec=json.loads((a.selected_study/(lang+'-review.json')).read_text())
        if selected_spec['adapter_sha256']!=file_sha(a.selected/(lang+'-selected.safetensors')):
            raise ValueError('Selected adapter binding differs')
        head=make_head({k:new['heads.'+lang+'.'+v] for k,v in
                       {'0.weight':'first.weight','0.bias':'first.bias','3.weight':'last.weight','3.bias':'last.bias'}.items()})
        component=frozen.models[lang];engine=PersistentPrefixEngine(*component._args);mx=engine.mx
        class CaptureHead(nn.Module):
            def __init__(self,inner):super().__init__();self.inner=inner;self.last_features=None;self.last_logits=None
            def __call__(self,features):
                values=self.inner(features);mx.eval(features,values)
                self.last_features=np.asarray(features).copy();self.last_logits=np.asarray(values).copy()
                return values
        tap=CaptureHead(engine.model.heads[lang]);engine.model.heads[lang]=tap
        records=[(name,c,point) for name,rows in cases.items() for c in rows if c['language']==lang for point in c['checkpoints']]
        x=np.lib.format.open_memmap(a.private/(lang+'-features.npy'),mode='w+',dtype=np.float32,shape=(len(records),2560))
        old_z=np.empty((len(records),sum(SIZES)+2),dtype=np.float32);new_z=np.empty_like(old_z)
        mappings=[];cache={}
        for i,(name,c,point) in enumerate(records):
            payload=prefix(c,point['focus']);view,inference=prepare_memory(payload,selector,1200,preserve_complete=True)
            if view['mandatory_overflow'] or not inference['turns'] or len(inference['turns'])>12:
                raise ValueError('Candidate context is unsupported; retain failed study')
            key=input_identity(inference);tick=time.perf_counter()
            if key in cache:features,old_logits,score=cache[key]
            else:
                score=engine.scores([inference],component._spec)[0]
                if score is None:raise ValueError('Model token envelope exceeded')
                features=tap.last_features[0].copy();old_logits=tap.last_logits[0].copy()
                cache[key]=(features,old_logits,score)
            values=head(mx.array(features[None,:]));mx.eval(values);z=np.asarray(values)[0].copy()
            candidate_score=_score(z,selected_spec)
            x[i]=features;old_z[i]=old_logits;new_z[i]=z
            mappings.append({'checkpoint_id':checkpoint_id(c,point),'population':name,
                             'input_digest':input_identity(payload),'inference_input_digest':key})
            if name=='broad':
                if inference!=payload:raise ValueError('Cannot claim v9 parity for a changed broad-regression input')
                original=original_scores[(lang,c['original_group'],c['original_case_id'])]
                error=abs(score-original['score']);max_error=max(max_error,error)
                if error>1e-5 or int(score>=component._spec['threshold'])!=original['predicted']:
                    raise ValueError('Frozen v9 reference parity failed')
            core={'checkpoint_id':checkpoint_id(c,point),'language':lang,'focus':point['focus'],
                  'input_digest':input_identity(payload),'inference_input_digest':key,
                  'evidence':[{k:s[k] for k in ('turn','start','end')} for s in view['spans']],
                  'latency_seconds':None,'qualified':False,'establishes_safety':False,'permits_external_action':False}
            for label,value,logits,spec in (('baseline',score,old_logits,component._spec),('candidate',candidate_score,z,selected_spec)):
                output[name][label].append({**core,'score':value,'predicted':int(value>=spec['threshold']),
                    'interpretation_set':{'hypotheses':[world_from_logits(logits)],'reason':'UNCALIBRATED_ARGMAX_DIAGNOSTIC',
                        'coverage_assumptions_verified':False,'outside_ontology_ruled_out':False}})
            done+=1
            if done%100==0 or i+1==len(records):
                x.flush();progress={'done':done,'language':lang,'population':name,'seconds':time.monotonic()-started,
                    'max_v9_score_error':max_error}
                (a.study/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
            mx.clear_cache()
        x.flush();del x
        np.save(a.private/(lang+'-original-logits.npy'),old_z,allow_pickle=False)
        np.save(a.private/(lang+'-candidate-logits.npy'),new_z,allow_pickle=False)
        write_jsonl(a.private/(lang+'-rows.jsonl'),mappings)
        for name,conditions in output.items():
            for condition,rows in conditions.items():write_jsonl(a.study/(name+'-'+condition+'.jsonl'),rows)
        del engine,head,tap,cache
        import gc
        gc.collect()
    report={'complete':True,'populations':{},'seconds':time.monotonic()-started,'max_v9_score_error':max_error,
            'fitting_performed':False,'stable_release_ready':False,
            'features':{lang:{suffix:file_sha(a.private/(lang+'-'+suffix)) for suffix in
                ('features.npy','original-logits.npy','candidate-logits.npy','rows.jsonl')} for lang in ('ro','uk')}}
    for name,conditions in output.items():
        report['populations'][name]={**{label:evaluate(cases[name],rows) for label,rows in conditions.items()},
                                     'comparison':paired_compare(cases[name],conditions['baseline'],conditions['candidate'])}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{label:{lang:g['strict_accuracy'] for lang,g in r[label]['groups'].items()} for label in ('baseline','candidate')}
                      for name,r in report['populations'].items()},indent=2))


if __name__=='__main__':main()
