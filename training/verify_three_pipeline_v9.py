"""Actual installed three-model inference, compared with the fixed study80 screen."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['root','bundle','installed-root','wheel','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bundle-sha256',required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    import roguard.model_bundle as module
    if a.installed_root.resolve() not in Path(module.__file__).resolve().parents:raise ValueError('Not an installed runtime')
    for path,digest in json.loads((a.root/'research/v9-stabilization/installed-pipeline-84/source-freeze.json').read_text()).items():
        if sha(a.root/path)!=digest:raise ValueError('Source freeze differs: '+path)
        if path.startswith('src/roguard/') and sha(a.installed_root/path.removeprefix('src/'))!=digest:
            raise ValueError('Installed source differs: '+path)
    model=module.load_bundle(a.bundle,expected_sha256=a.bundle_sha256)
    rows=[json.loads(line) for line in (a.root/'research/v8-stabilization/instruction-sft-01/screen.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['language']=='ro']
    reference=[json.loads(line) for line in (a.root/'research/v9-stabilization/installed-three-expert-80/screen/predictions.jsonl').read_text().splitlines()]
    if len(rows)!=200 or [r['case_id'] for r in rows]!=[r['case_id'] for r in reference]:raise ValueError('Screen coverage differs')
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();results=[];errors=[];changes=0
    for offset in range(0,len(rows),20):
        batch=rows[offset:offset+20]
        values=model.predict_many([dict(language=r['language'],turns=r['turns']) for r in batch],focuses=[r['focus'] for r in batch])
        for r,value,old in zip(batch,values,reference[offset:offset+20]):
            if value['score'] is None:raise ValueError('Unexpected abstention')
            changes+=value['predicted']!=old['predicted'];errors.append(abs(value['score']-old['score']))
            results.append(dict(case_id=r['case_id'],gold=old['gold'],kind=old['kind'],**value))
        print('completed',len(results),'of200',flush=True)
    import importlib.metadata as metadata
    report=dict(rows=len(results),decision_changes=changes,maximum_score_error=max(errors),parity_pass=changes==0 and max(errors)<=.001,
        actual_models_loaded=True,cached_component_outputs_used=False,seconds=time.monotonic()-start,
        bundle_sha256=a.bundle_sha256,wheel_sha256=sha(a.wheel),
        versions={k:metadata.version(k) for k in ['torch','transformers','mlx','mlx-lm']},
        reserved_probes_used=False,stable_release_ready=False,independent_accuracy_evaluation=False)
    (a.output/'predictions.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in results))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not report['parity_pass']:raise SystemExit('Actual pipeline parity failed')


if __name__=='__main__':main()
