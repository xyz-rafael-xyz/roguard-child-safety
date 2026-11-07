"""Run frozen v9, retrieval-wrapped v9, or the portable v10 research candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from roguard.assurance import digest
from roguard.trajectory_data import read_jsonl, write_jsonl, prefix, input_identity
from roguard.trajectory_evaluation import checkpoint_id, evaluate
from roguard.trajectory_memory import select
from roguard.trajectory_model import TrajectoryModel


class LocalFrozenV9:
    """Use existing research weights after matching the released adapter digest."""
    def __init__(self, base, candidate, staging):
        from roguard.gemma4_review import Gemma4ReviewModel
        from roguard.gemma4_contract import BASE_MODEL, BASE_REVISION, POLICY_SHA256
        from roguard.assurance import policy_digest
        from roguard.trajectory_data import file_sha
        self.models={}
        staging.mkdir(parents=True,exist_ok=False)
        # Compare both languages with the qualified public download manifest.
        from importlib import resources
        catalog=json.loads(resources.files('roguard').joinpath('model-download-v9-gemma4-20261019.json').read_text())
        for lang in ('ro','uk'):
            spec=json.loads((candidate/'actual'/f'{lang}-review.json').read_text())
            adapter=candidate/'actual'/f'{lang}-selected.safetensors'
            matches=[f for f in catalog['files'] if f['path'].endswith(f'{lang}-adapter.safetensors')]
            if len(matches)!=1 or file_sha(adapter)!=matches[0]['sha256'] or spec['checkpoint_sha256']!=matches[0]['sha256']:
                raise ValueError('Local adapter differs from published v9')
            metadata={'format':'roguard-gemma4-tail-review-v1','language':lang,'base_model':BASE_MODEL,
                      'base_revision':BASE_REVISION,'policy_sha256':POLICY_SHA256,'policy_digest':policy_digest(),
                      'adapter_sha256':spec['checkpoint_sha256'],'mode':spec['mode'],'couplings':spec['couplings'],
                      'threshold':spec['threshold'],'boundary':spec['boundary'],'lora':spec['lora']}
            from roguard.gemma4_review import FORMAT
            metadata['format']=FORMAT
            path=staging/f'{lang}-review.json';path.write_text(json.dumps(metadata,indent=2)+'\n')
            public=[f for f in catalog['files'] if f['path'].endswith(f'{lang}-review.json')]
            if len(public)!=1 or file_sha(path)!=public[0]['sha256']:
                raise ValueError('Local review parameters differ from published v9')
            self.models[lang]=Gemma4ReviewModel(path,adapter,base,expected_sha256=file_sha(path))
        self.active=None

    def predict(self,payload):
        lang=payload['language']
        if self.active is not None and self.active!=lang:
            self.models[self.active]._engine=None
            import gc
            gc.collect()
        self.active=lang
        return self.models[lang].predict(payload)


def run(cases, *, model=None, calibration=None, strategy='learned', backend='student', v9_model=None, plan=False):
    predictions=[]
    for case in cases:
        for point in case['checkpoints']:
            payload=prefix(case,point['focus'])
            started=time.perf_counter()
            if backend=='student':
                result=model.predict(payload,calibration=calibration,strategy=strategy,plan=plan)
            else:
                view=None
                if backend=='v9-retrieval':
                    view=select(payload,strategy='lexical',budget=2000)
                    inference={'language':payload['language'],'turns':[]}
                    for s in view['spans']:
                        if inference['turns'] and inference['turns'][-1].get('_turn')==s['turn']:
                            inference['turns'][-1]['text']+=' '+s['text']
                        else:inference['turns'].append({'role':s['role'],'text':s['text'],'_turn':s['turn']})
                    if len(inference['turns'])>12:
                        # Respect v9's 12-turn contract; use lexical overlap to
                        # choose eleven history turns, retaining the focal turn.
                        import re
                        words=set(re.findall(r'\w+',inference['turns'][-1]['text'].casefold()))
                        ranked=sorted(range(len(inference['turns'])-1),key=lambda i:(
                            -len(words & set(re.findall(r'\w+',inference['turns'][i]['text'].casefold()))),-i))[:11]
                        inference['turns']=[inference['turns'][i] for i in sorted(ranked)]+inference['turns'][-1:]
                    kept_turns={t['_turn'] for t in inference['turns']}
                    view['spans']=[s for s in view['spans'] if s['turn'] in kept_turns]
                    for t in inference['turns']:t.pop('_turn')
                else:inference=payload
                if view is not None and view['mandatory_overflow']:
                    raw={'predicted':None,'reason':'MEMORY_BUDGET_EXCEEDED'}
                else:
                    try:
                        raw=v9_model.predict(inference)
                    except ValueError as error:
                        # The frozen v9 input contract can reject long prefixes
                        # before tokenization. Preserve that failure as abstention.
                        from roguard.interpreter import prepare_input
                        try:prepare_input(inference)
                        except ValueError:
                            raw={'predicted':None,'reason':'V9_INPUT_REJECTION','error_type':type(error).__name__}
                        else:raise
                result={**raw,'language':payload['language'],'focus':point['focus'],
                        'input_digest':input_identity(payload),'backend':backend,
                        'inference_input_digest':input_identity(inference) if inference['turns'] else None,
                        'v9_raw_result_digest':digest(raw),'establishes_safety':False,'permits_external_action':False}
                if view is not None:
                    result['evidence']=[{k:v for k,v in s.items() if k in ('turn','start','end')} for s in view['spans']]
                elif raw['predicted'] is not None:
                    from roguard.trajectory_memory import units
                    result['evidence']=[{k:s[k] for k in ('turn','start','end')} for s in units(payload)]
            result['latency_seconds']=time.perf_counter()-started
            result['checkpoint_id']=checkpoint_id(case,point)
            predictions.append(result)
    return predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['student','v9','v9-retrieval','full-context'],default='student')
    p.add_argument('--model',type=Path);p.add_argument('--calibration',type=Path)
    p.add_argument('--strategy',choices=['learned','recent','lexical'],default='learned');p.add_argument('--plan',action='store_true')
    p.add_argument('--v9-base',type=Path);p.add_argument('--v9-candidate',type=Path)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite an evaluation')
    cases=read_jsonl(a.cases)
    started=time.perf_counter()
    model=TrajectoryModel(a.model) if a.backend=='student' else None
    v9=None
    if a.backend!='student':
        if a.v9_base:
            if not a.v9_candidate:raise ValueError('Local v9 baseline requires its qualified candidate directory')
            v9=LocalFrozenV9(a.v9_base,a.v9_candidate,a.output.with_name(a.output.name+'-binding'))
        else:
            from roguard.model_install import cache_directory,profile,install
            from roguard.entailment_bundle import load_bundle
            raw,sha=profile('v9-gemma4-20261019')
            path,bundle_sha=install(raw,sha,cache_directory(),download=False)
            v9=load_bundle(path,expected_sha256=bundle_sha)
    calibration=json.loads(a.calibration.read_text()) if a.calibration else None
    loading=time.perf_counter()-started
    if a.backend=='full-context':
        from roguard.trajectory_neural import FullContextComparator
        if not isinstance(v9,LocalFrozenV9):raise ValueError('Full-context comparator requires local frozen components')
        comparator=FullContextComparator(v9.models)
        predictions=[]
        for c in cases:
            for point in c['checkpoints']:
                tick=time.perf_counter();r=comparator.predict(prefix(c,point['focus']))
                r['checkpoint_id']=checkpoint_id(c,point);r['latency_seconds']=time.perf_counter()-tick
                predictions.append(r)
    else:predictions=run(cases,model=model,calibration=calibration,strategy=a.strategy,backend=a.backend,v9_model=v9,plan=a.plan)
    report=evaluate(cases,predictions)
    report['artifact_load_seconds']=loading
    report['backend']=a.backend;report['strategy']=a.strategy if a.backend=='student' else None
    report['latency_note']='First prediction includes any lazy model startup; no steady-state claim.'
    a.output.mkdir(parents=True)
    write_jsonl(a.output/'predictions.jsonl',predictions)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
