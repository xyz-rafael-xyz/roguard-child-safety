"""Run each installed neural component in its own environment, then compose.

Invoke native-ro, native-uk and interpreter serially in separate processes.
The fusion stage loads only their newly generated, input-bound reports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

if __package__:
    from .reserved_fusion_v9 import qualify
    from .reserved_metrics_v9 import counts, input_key, unique_records
    from .verify_reserved_native_v9 import PROBES
else:
    from reserved_fusion_v9 import qualify
    from reserved_metrics_v9 import counts, input_key, unique_records
    from verify_reserved_native_v9 import PROBES


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('native-ro','native-uk','interpreter','fusion'))
    for name in ('root','installed-root','plan','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--base',type=Path)
    a=p.parse_args()
    if a.stage!='fusion' and (a.base is None or not a.base.is_dir()):
        p.error('Neural stages require an existing explicit local --base directory')
    if sha(a.plan)!=a.plan_sha256:raise ValueError('Reserved candidate plan changed')
    plan=json.loads(a.plan.read_text())
    if plan.get('format')!='roguard-reserved-fusion-plan-v1' or set(plan['candidates'])!={'ro','uk'}:
        raise ValueError('Incomplete reserved fusion plan')
    checked=qualify(a.root)
    if {k:v for k,v in checked.items() if k!='recorded_utc'}!={k:v for k,v in plan.items() if k!='recorded_utc'}:
        raise ValueError('Qualification or verifier changed after candidate freeze')
    for relative,expected in plan['evidence_sha256'].items():
        if sha(a.root/relative)!=expected:raise ValueError('Frozen selection evidence changed')
    for relative,expected in plan['runtime_source_sha256'].items():
        path=a.installed_root/Path(relative).relative_to('src')
        if sha(path)!=expected:raise ValueError('Installed runtime source differs')
    import roguard.evidence as evidence
    if a.installed_root.resolve() not in Path(evidence.__file__).resolve().parents:
        raise ValueError('Runtime import must come from the isolated wheel')
    rows=[]
    for name,(relative,expected) in PROBES.items():
        path=a.root.parent/'v8-stabilization'/relative
        if sha(path)!=expected:raise ValueError('Reserved inputs changed')
        members=read(path)
        if len({r['case_id'] for r in members})!=len(members):raise ValueError('Duplicate probe identity')
        for row in members:
            gold=row.get('review')
            if gold is None:gold=int(evidence.disposition({k:{'label':v} for k,v in row['attributes'].items()})[0]!='CONTINUE')
            rows.append(dict(row,probe=name,gold=gold))
    a.output.mkdir(parents=True,exist_ok=True)
    target=a.output/(a.stage+'-predictions.jsonl')
    if target.exists():raise ValueError('This reserved stage already has predictions')
    start=time.monotonic();results=[];input_reports={}
    def append(row,value):
        if value['input_digest']!=input_key(row):raise ValueError('Reserved input binding differs')
        result=dict(probe=row['probe'],case_id=row['case_id'],gold=row['gold'],**value)
        results.append(result);stream.write(json.dumps(result)+'\n');stream.flush()
    with target.open('x') as stream:
        if a.stage.startswith('native-'):
            import torch
            from roguard.native_review import NativeReviewModel
            torch.set_num_threads(4)
            language=a.stage[-2:];entry=plan['candidates'][language]
            model=NativeReviewModel(a.root/entry['native_artifact'],a.root/entry['checkpoint'],a.base,
                a.root/'native-full-22-base-manifest.json',expected_artifact_sha256=entry['native_artifact_sha256'],device='cpu')
            members=[r for r in rows if r['language']==language]
            for offset in range(0,len(members),128):
                batch=members[offset:offset+128]
                values=model.predict_many([dict(language=r['language'],turns=r['turns']) for r in batch],
                    focuses=[r.get('focus',len(r['turns'])-1) for r in batch])
                if len(values)!=len(batch):raise ValueError('Missing native outputs')
                for row,value in zip(batch,values):append(row,value)
        elif a.stage=='interpreter':
            import mlx.core as mx
            from roguard.interpreter import InterpreterModel
            mx.set_cache_limit(512*1024*1024)
            model=InterpreterModel(a.root/plan['interpreter_archive'],a.base,
                expected_archive_sha256=plan['interpreter_archive_sha256'])
            for i,row in enumerate(rows):
                append(row,model.predict(dict(language=row['language'],turns=row['turns']),focus=row.get('focus',len(row['turns'])-1)))
                if (i+1)%25==0:print(i+1,'of',len(rows),flush=True)
        else:
            from roguard.review_fusion import ReviewFusion
            indexed={}
            for component in ('native-ro','native-uk','interpreter'):
                path=a.output/(component+'-predictions.jsonl')
                report_path=a.output/(component+'-report.json')
                report=json.loads(report_path.read_text())
                if (report['plan_sha256']!=a.plan_sha256 or report.get('complete') is not True
                        or report['predictions_sha256']!=sha(path)):
                    raise ValueError('Reserved component evidence differs')
                input_reports[component]=sha(report_path)
                values=read(path);index={(r['probe'],r['case_id']):r for r in values}
                expected={(r['probe'],r['case_id']) for r in rows if component=='interpreter' or r['language']==component[-2:]}
                if len(index)!=len(values) or set(index)!=expected:raise ValueError('Reserved component coverage differs')
                indexed[component]=index
            models={language:ReviewFusion(a.root/entry['artifact'],expected_artifact_sha256=entry['artifact_sha256'])
                    for language,entry in plan['candidates'].items()}
            for row in rows:
                key=(row['probe'],row['case_id']);language=row['language']
                n,s=indexed['native-'+language][key],indexed['interpreter'][key]
                if not n['gold']==s['gold']==row['gold']:raise ValueError('Reserved labels differ')
                append(row,models[language].predict(dict(language=language,turns=row['turns']),n,s,
                    focus=row.get('focus',len(row['turns'])-1)))
    report=dict(format='roguard-reserved-fusion-stage-v1',stage=a.stage,complete=True,rows=len(results),
        seconds=time.monotonic()-start,plan_sha256=a.plan_sha256,predictions_sha256=sha(target),
        verifier_sha256=sha(__file__),input_component_report_sha256=input_reports,
        neural_models_loaded=a.stage!='fusion',actual_installed_neural_outputs_used=True,
        synthetic_only=True,analyst_blind=False,independently_annotated=False,stable_release_ready=False)
    if a.stage=='fusion':
        report['metrics']={name:{language:counts([r for r in results if r['probe']==name and r['language']==language])
            for language in ('ro','uk')} for name in PROBES}
        unique=unique_records(results)
        report['unique_inputs']=len(unique)
        report['unique_input_counts']={language:counts([r for r in unique if r['language']==language]) for language in ('ro','uk')}
        report['passes']=all(g['passes'] for dataset in report['metrics'].values() for g in dataset.values())
        report['independent_observations']=False
        report['limitation']='Related authored probes; counts do not establish population accuracy or product readiness.'
    (a.output/(a.stage+'-report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
