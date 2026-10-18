"""Replay selected semantic heads through an isolated installed wheel."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

if __package__:
    from .reserved_metrics_v9 import counts,input_key
else:
    from reserved_metrics_v9 import counts,input_key


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','base','installed-root','wheel','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--language',choices=('ro','uk'),required=True);a=p.parse_args()
    import torch
    import roguard.native_fields as runtime
    from roguard.evidence import disposition
    torch.set_num_threads(4)
    if a.installed_root.resolve() not in Path(runtime.__file__).resolve().parents:
        raise ValueError('Semantic runtime must come from isolated wheel')
    frozen=json.loads((a.root/'native-fields-43/source-freeze.json').read_text())
    if sha(runtime.__file__)!=frozen['src/roguard/native_fields.py']:
        raise ValueError('Installed semantic runtime differs from pre-fit freeze')
    study=a.root/'native-fields-43'/a.language;parent=a.root/('native-full-22-'+a.language)
    selection=json.loads((study/'selection.json').read_text());audit=json.loads((study/'audit.json').read_text())
    if (audit['selection_sha256']!=sha(study/'selection.json')
            or selection['artifact_sha256']!=sha(study/'selected-heads.json.gz')):
        raise ValueError('Selected head evidence changed')
    model=runtime.NativeFieldsModel(study/'selected-heads.json.gz',parent/'review.json.gz',
        parent/'selected-model.safetensors',a.base,a.root/'native-full-22-base-manifest.json',
        expected_heads_sha256=selection['artifact_sha256'],device='cpu')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(format='roguard-installed-native-fields-replay-v1',language=a.language,
        artifact_sha256=selection['artifact_sha256'],selection_sha256=sha(study/'selection.json'),
        native_artifact_sha256=sha(parent/'review.json.gz'),wheel_sha256=sha(a.wheel),
        runtime_source_sha256=sha(runtime.__file__),installed_module=str(Path(runtime.__file__).resolve()),
        probability_tolerance=1e-4,permitted_decision_changes=0,splits={},synthetic_only=True,
        reserved_probes_used=False,stable_release_ready=False)
    for split in ('dev','cal'):
        data=parent/(split+'.jsonl.gz');rows=[json.loads(l) for l in gzip.decompress(data.read_bytes()).splitlines()]
        reference_path=study/(split+'-predictions.jsonl');reference=[json.loads(l) for l in reference_path.read_text().splitlines()]
        if [r['case_id'] for r in rows]!=[r['case_id'] for r in reference]:raise ValueError('Reference coverage differs')
        results=[];start=time.monotonic()
        for offset in range(0,len(rows),128):
            batch=rows[offset:offset+128]
            values=model.predict_many([dict(language=r['language'],turns=r['turns']) for r in batch],focuses=[r['focus'] for r in batch])
            for row,value in zip(batch,values):
                if value['input_digest']!=input_key(row):raise ValueError('Input binding differs')
                gold=int(disposition({f:{'label':v} for f,v in row['attributes'].items()})[0]!='CONTINUE')
                results.append(dict(case_id=row['case_id'],gold=gold,
                    kind='conversational' if row.get('source')=='v7_training_frame_projection' else 'formal',**value))
            print(split,len(results),'of',len(rows),flush=True)
        if len(results)!=len(rows) or any(r['gold']!=old['gold'] for r,old in zip(results,reference)):
            raise ValueError('Output coverage or policy labels differ')
        changes=sum(r['predicted']!=old['predicted'] for r,old in zip(results,reference))
        abstentions=sum(r['score'] is None for r in results)
        error=max((abs(r['score']-old['score']) for r,old in zip(results,reference) if r['score'] is not None),default=None)
        field_error=max((abs(x-y) for r,old in zip(results,reference) if r['field_probabilities'] is not None
            for field,v in r['field_probabilities'].items() for x,y in zip(v,old['field_probabilities'][field])),default=None)
        metrics={kind:counts([r for r in results if r['kind']==kind]) for kind in ('formal','conversational')}
        path=a.output/(split+'-predictions.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in results))
        report['splits'][split]=dict(rows=len(rows),seconds=time.monotonic()-start,decision_changes=changes,
            abstentions=abstentions,maximum_score_error=error,maximum_field_error=field_error,counts=metrics,
            parity_pass=bool(not changes and not abstentions and error is not None and error<=1e-4 and field_error<=1e-4),
            accuracy_gates_pass=all(v['passes'] for v in metrics.values()),input_sha256=sha(data),
            reference_sha256=sha(reference_path),predictions_sha256=sha(path))
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,parity_pass=all(v['parity_pass'] for v in report['splits'].values()),
        accuracy_gates_pass=all(v['accuracy_gates_pass'] for v in report['splits'].values()))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not report['parity_pass']:raise SystemExit('Installed semantic-head parity failed')


if __name__=='__main__':main()
