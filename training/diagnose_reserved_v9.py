"""Capture component evidence for the now-exposed study115 regression cases."""
import argparse
import json
from pathlib import Path
import time
from training.verify_reserved_native_v9 import PROBES, digest
from training.verify_candidate_pipeline_v9 import Capture


def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',type=Path,required=True);a=p.parse_args()
    root=Path.cwd();study=root/'research/v9-stabilization/exposed-diagnosis-116'
    for name, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(root/name)!=expected:raise ValueError('Frozen source changed: '+name)
    from roguard.consensus_bundle import load_bundle
    import torch
    torch.set_num_threads(4)
    meta=json.loads(a.metadata.read_text());bundle=Path(meta['bundle'])
    if digest(bundle)!=meta['bundle_sha256']:raise ValueError('Bundle changed')
    specification=json.loads(bundle.read_text())
    reference=[json.loads(l) for l in (root/'research/v9-stabilization/bilingual-consensus-115/actual/predictions.jsonl').read_text().splitlines()]
    source={}
    for name,(filename,expected) in PROBES.items():
        path=root/'research/v8-stabilization'/filename
        if digest(path)!=expected:raise ValueError('Probe changed')
        for line in path.read_text().splitlines():
            row=json.loads(line);source[name,row['case_id']]=row
    output=study/'actual';output.mkdir(exist_ok=False);summaries={}
    for language in ('ro','uk'):
        entry=specification['languages'][language]
        model=load_bundle(bundle.parent/entry['path'],expected_sha256=entry['sha256'])
        pipeline=model.base if hasattr(model,'base') else model
        captures={name:Capture(getattr(pipeline,name)) for name in ('native','semantic','interpreter')}
        for name,value in captures.items():setattr(pipeline,name,value)
        seen=set();refs=[];rows=[]
        for r in reference:
            if r['language']==language and r['input_digest'] not in seen:
                seen.add(r['input_digest']);refs.append(r);rows.append(source[r['probe'],r['case_id']])
        predictions=[];start=time.monotonic()
        for offset in range(0,len(rows),20):
            batch=rows[offset:offset+20]
            predictions.extend(model.predict_many([dict(language=language,turns=r['turns']) for r in batch],focuses=[r.get('focus',len(r['turns'])-1) for r in batch]))
            print(language,len(predictions),'/',len(rows),flush=True)
        error=max(abs(r['score']-v['score']) for r,v in zip(refs,predictions))
        changes=sum(r['predicted']!=v['predicted'] for r,v in zip(refs,predictions))
        if error>.001 or changes:raise ValueError('Diagnosis replay differs')
        for name,capture in captures.items():
            if len(capture.values)!=len(rows):raise ValueError('Missing component outputs')
            (output/(language+'-'+name+'.jsonl')).write_text(''.join(json.dumps(dict(v,case_id=r['case_id'],gold=r['gold']))+'\n' for v,r in zip(capture.values,refs)))
        summaries[language]=dict(rows=len(rows),seconds=time.monotonic()-start,maximum_score_error=error,decision_changes=changes)
        del model,pipeline,captures
        import gc
        gc.collect()
    (output/'report.json').write_text(json.dumps(dict(complete=True,groups=summaries,independent_accuracy_evaluation=False,previously_reserved_now_exposed=True,synthetic_only=True),indent=2)+'\n')


if __name__=='__main__':main()
