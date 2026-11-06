"""Fresh neural inference for changed attention adapters on exposed regressions."""
import argparse
import json
from pathlib import Path
import re
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_review import _score
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl, prefix, input_identity
from roguard.trajectory_evaluation import evaluate, paired_compare, checkpoint_id
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import FieldEngine, prepare_memory
from training.evaluate_semantic_readout_v10 import world_from_logits


class CachedEngine(FieldEngine):
    cache_instructions = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base','candidate','selected','selected_study','reference_study','reference_private','selector','study'):
        p.add_argument('--'+key.replace('_','-'),dest=key,type=Path,required=True)
    a=p.parse_args()
    if a.study.exists():raise ValueError('Refusing to overwrite changed-attention evaluation')
    selection=json.loads((a.selected_study/'report.json').read_text())
    reference=json.loads((a.reference_study/'report.json').read_text())
    if not selection['complete'] or not reference['complete']:raise ValueError('Incomplete parent study')
    reference_protocol=json.loads((a.reference_study/'protocol.json').read_text())
    if reference_protocol['format']!='roguard-v10-semantic-regression-1':raise ValueError('Unknown original baseline')
    for lang,artifacts in reference['features'].items():
        for suffix,expected in artifacts.items():
            if file_sha(a.reference_private/(lang+'-'+suffix))!=expected:raise ValueError('Original reference differs')
    paths={'broad':Path('research/v10-evidence/broad-v9-regression/cases.jsonl'),
        **{name:Path('research/v10-evidence/data')/(name+'.jsonl') for name in ('dev','challenge','v9-regression')}}
    for name,path in paths.items():
        if file_sha(path)!=reference_protocol['inputs'][name]:raise ValueError('Evaluation population differs')
    selector=TrajectoryModel(a.selector)
    if selector.identity!=reference_protocol['selector_sha256']:raise ValueError('Baseline memory selector differs')
    a.study.mkdir(parents=True)
    frozen=LocalFrozenV9(a.base,a.candidate,a.study/'v9-binding')
    protocol={'format':'roguard-semantic-tail-regression-1','source_sha256':file_sha(Path(__file__)),
        'selection_report_sha256':file_sha(a.selected_study/'report.json'),
        'original_reference_report_sha256':file_sha(a.reference_study/'report.json'),
        'inputs':{k:file_sha(v) for k,v in paths.items()},'selector_sha256':selector.identity,
        'selected_adapters':{lang:file_sha(a.selected/(lang+'-selected.safetensors')) for lang in ('ro','uk')},
        'fresh_candidate_neural_inference':True,'frozen_encoder_changed':True,
        'baseline':'Previously verified v9 with the same uniform memory view; broad and original 600 inputs unchanged',
        'baseline_reused':True,'end_to_end_latency_measured':False,'fitting_performed':False,
        'threshold_selection_performed':False,'independent_test':False,'stable_release_ready':False}
    (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    import mlx.nn as nn
    from safetensors.numpy import load_file
    cases={name:read_jsonl(path) for name,path in paths.items()}
    baseline={name:read_jsonl(a.reference_study/(name+'-baseline.jsonl')) for name in paths}
    originals={name:{r['checkpoint_id']:r for r in rows} for name,rows in baseline.items()}
    outputs={name:[] for name in paths}; started=time.monotonic();done=0;forwards=0
    for lang in ('ro','uk'):
        original=load_file(a.candidate/'actual'/(lang+'-selected.safetensors'))
        selected=load_file(a.selected/(lang+'-selected.safetensors'))
        if set(original)!=set(selected):raise ValueError('Candidate tensor inventory differs')
        for name in original:
            allowed=name.startswith('heads.'+lang+'.') or re.fullmatch(r'base\.model\.layers\.(\d+)\.self_attn\.(q_proj|o_proj)\.lora_[ab]',name)
            if not allowed and not np.array_equal(original[name],selected[name]):raise ValueError('Unregistered candidate weight change')
        spec=json.loads((a.selected_study/(lang+'-review.json')).read_text())
        if spec['adapter_sha256']!=file_sha(a.selected/(lang+'-selected.safetensors')):raise ValueError('Selected runtime binding differs')
        engine=CachedEngine(a.base,a.selected/(lang+'-selected.safetensors'),spec['adapter_sha256']);mx=engine.mx
        class Tap(nn.Module):
            def __init__(self,inner):super().__init__();self.inner=inner
            def __call__(self,h):
                z=self.inner(h);mx.eval(z);self.logits=np.asarray(z)[0].copy();return z
        tap=Tap(engine.model.heads[lang]);engine.model.heads[lang]=tap
        mappings=read_jsonl(a.reference_private/(lang+'-rows.jsonl'))
        positions={(r['population'],r['checkpoint_id']):i for i,r in enumerate(mappings)}
        old_logits=np.load(a.reference_private/(lang+'-original-logits.npy'),mmap_mode='r')
        identity=digest({'adapter':spec['adapter_sha256'],'spec':spec,'selector':selector.identity,'memory_budget':1200})
        memo={}
        for population,rows in cases.items():
            for case in (r for r in rows if r['language']==lang):
                for point in case['checkpoints']:
                    key=checkpoint_id(case,point);old=originals[population][key]
                    payload=prefix(case,point['focus']);view,inference=prepare_memory(payload,selector,1200,preserve_complete=True)
                    inference_digest=input_identity(inference)
                    if view['mandatory_overflow'] or len(inference['turns'])>12:raise ValueError('Unsupported memory view')
                    if old['input_digest']!=input_identity(payload) or old['inference_input_digest']!=inference_digest:
                        raise ValueError('Baseline inference context differs')
                    position=positions[(population,key)]
                    old_score=_score(old_logits[position],frozen.models[lang]._spec)
                    if abs(old_score-old['score'])>1e-7 or int(old_score>=frozen.models[lang]._spec['threshold'])!=old['predicted']:
                        raise ValueError('Baseline prediction differs from verified original logits')
                    if inference_digest not in memo:
                        score=engine.scores([inference],spec)[0]
                        if score is None:raise ValueError('Candidate model token envelope exceeded')
                        memo[inference_digest]=(score,tap.logits.copy());forwards+=1
                    score,logits=memo[inference_digest]
                    outputs[population].append({**old,'model_identity':identity,'score':score,
                        'predicted':int(score>=spec['threshold']),'field_logits':logits.tolist(),
                        'interpretation_set':{'hypotheses':[world_from_logits(logits)],
                            'reason':'UNCALIBRATED_ARGMAX_DIAGNOSTIC','coverage_assumptions_verified':False,
                            'outside_ontology_ruled_out':False}})
                    done+=1
                    if done%100==0:
                        progress={'done':done,'fresh_neural_forwards':forwards,'language':lang,
                                  'population':population,'seconds':time.monotonic()-started}
                        (a.study/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
        for name,rows in outputs.items():write_jsonl(a.study/(name+'-candidate.jsonl'),rows)
        del engine,tap,memo
        import gc
        gc.collect();mx.clear_cache()
    report={'complete':True,'populations':{},'fresh_neural_forwards':forwards,'seconds':time.monotonic()-started,
            'independent_test':False,'stable_release_ready':False,'fitting_performed':False}
    for name,rows in cases.items():
        report['populations'][name]={'baseline':evaluate(rows,baseline[name]),'candidate':evaluate(rows,outputs[name]),
                                    'comparison':paired_compare(rows,baseline[name],outputs[name])}
    report['prediction_sha256']={name:file_sha(a.study/(name+'-candidate.jsonl')) for name in cases}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{lang:g['strict_accuracy'] for lang,g in p['candidate']['groups'].items()}
                      for name,p in report['populations'].items()},indent=2))


if __name__=='__main__':main()
