"""Paired, text-free metric reports for frozen v7.1 context candidates.

A benign alert counts REVIEW (including abstention), not just a positive argmax.
Previous v7 probes are regressions. Only --suite fresh uses the new frozen set.
"""
import argparse
import hashlib
import json
from pathlib import Path
from roguard.context import ContextModel, analyze_context, render


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--legacy-policy',action='store_true')
    parser.add_argument('--base',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--suite',choices=['regression','fresh','stress','session','final2','final3'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Do not overwrite experiment evidence')
    if args.suite in ('fresh','stress','final2','final3'):
        path=Path('research/v7-1-context/'+args.suite+'-test.jsonl')
        freeze=json.loads(path.with_name('test-freeze.json' if args.suite=='fresh' else args.suite+'-freeze.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=freeze['sha256']: raise ValueError('Changed frozen test')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
    elif args.suite=='session':
        from training.context_session_probe_v7 import generate
        rows=generate()
        for r in rows:
            r['signal']=int(r['kind']!='benign');r['frame']={'benign':'ORDINARY','personal':'SELF_CURRENT','quoted':'QUOTED','correction':'CORRECTION','uncertain':'UNCERTAIN'}[r['kind']]
    else:
        from training.context_factorial_final_v7 import generate
        rows=[r for r in generate() if r['surface']=='original']
    analyze=analyze_context
    policy_path=Path('src/roguard/context.py')
    if args.legacy_policy:
        import importlib.util
        policy_path=Path('research/v7-1-context/baseline-policy.py')
        spec=importlib.util.spec_from_file_location('roguard._baseline_context',policy_path)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy);analyze=legacy.analyze_context
    model=ContextModel('joint',args.base,args.device,artifact=args.artifact)
    record={'rules_sha256':None if args.legacy_policy else hashlib.sha256(Path('src/roguard/context_signals.py').read_bytes()).hexdigest(),'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),'legacy_policy':args.legacy_policy,'suite':args.suite,'synthetic_only':True,'independent_native_review':False,'model_sha256':model.identity,'cells':{},'outputs':[]}
    for row in rows:
        report=analyze({'language':row['language'],'turns':row['turns']},model)
        review=report['decision']=='REVIEW'
        kind='benign' if row['signal']==0 else 'personal' if row['frame'] in ('SELF_CURRENT','SELF_PAST') else 'quoted' if row['frame']=='QUOTED' else 'other_concern'
        cell=record['cells'].setdefault(row['language']+'/'+kind,dict(count=0,review=0))
        cell['count']+=1;cell['review']+=int(review)
        final=model.predict([render(row['turns'])],row['language'])[0]['heads']
        record['outputs'].append(dict(case_id=row['case_id'],language=row['language'],expected_frame=None if args.suite in ('session','final3') else row['frame'],expected_signal=row['signal'],predicted_frame=final['frame'],predicted_signal=final['signal'],decision=report['decision'],review_turn_indices=report['review_turn_indices'],evidence=report['evidence']))
    # Session rows carry decision-level labels only, not frame gold labels.
    from sklearn.metrics import f1_score
    record['metrics']={}
    for lang in (() if args.suite in ('session','final3') else ('ro','uk')):
        subset=[r for r in record['outputs'] if r['language']==lang]
        record['metrics'][lang]={task+'_macro_f1':float(f1_score([r['expected_'+task] for r in subset],[r['predicted_'+task]['argmax'] for r in subset],average='macro')) for task in ('frame','signal')}
    args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='outputs'},indent=2),flush=True)

if __name__=='__main__': main()
