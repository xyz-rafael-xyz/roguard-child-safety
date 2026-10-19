"""Replay the qualified Romanian tree combination over actual installed reports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]


def main():
    p=argparse.ArgumentParser()
    for name in ['root','installed-root','wheel','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path.append(str(a.root.parent.parent))
    import roguard.three_expert as runtime
    from training.three_expert_fusion_v9 import load_split
    from training.boosted_fusion_v9 import probability
    from training.reserved_fusion_v9 import require_counts
    from training.reserved_metrics_v9 import counts
    if a.installed_root.resolve() not in Path(runtime.__file__).resolve().parents:raise ValueError('Not an installed runtime')
    study=a.root/'installed-three-expert-80'
    for path,digest in json.loads((study/'source-freeze.json').read_text()).items():
        if sha(a.root.parent.parent/path)!=digest:raise ValueError('Source freeze changed: '+path)
    if sha(runtime.__file__)!=sha(a.root.parent.parent/'src/roguard/three_expert.py'):raise ValueError('Installed source differs')
    selected=a.root/'conservative-boosted-79/78/ro/selection.json'
    selection=json.loads(selected.read_text());audit=json.loads(selected.with_name('audit.json').read_text())
    for group in selection['development'].values():require_counts(group)
    require_counts(audit['calibration'])
    if not audit['counts_pass'] or audit['selection_sha256']!=sha(selected):raise ValueError('Candidate qualification differs')
    artifact=study/'fusion.json';export=json.loads((study/'export.json').read_text())
    model=runtime.ThreeExpertReview(artifact,expected_artifact_sha256=export['artifact_sha256'])
    if model._spec['selection_sha256']!=sha(selected) or model._spec['model']!=selection['model'] or model._spec['threshold']!=selection['threshold']:
        raise ValueError('Export differs from qualified selection')
    dirs=[a.root/'installed-native-29/ro',a.root/'installed-semantic-selection-72/ro',a.root/'installed-interpreter-30/screen']
    report_hashes={}
    for directory,key in zip(dirs,['native_artifact_sha256','field_artifact_sha256','interpreter_archive_sha256']):
        report=json.loads((directory/'report.json').read_text());identity=report.get('artifact_sha256',report.get('archive_sha256'))
        if report.get('parity_pass') is not True or identity!=model._spec[key]:raise ValueError('Component parity or identity differs')
        report_hashes[str(directory.relative_to(a.root))]=sha(directory/'report.json')
    def indexed(paths):
        rows=[v for path in paths for v in read(path)];out={v['case_id']:v for v in rows}
        if len(out)!=len(rows):raise ValueError('Duplicate component cases')
        return out
    native=indexed([dirs[0]/(s+'-predictions.jsonl') for s in ['dev','cal']])
    semantic=indexed([dirs[1]/(s+'-predictions.jsonl') for s in ['dev','cal']])
    interpreter=indexed([dirs[2]/'predictions.jsonl'])
    reference={}
    for split in ['dev','cal']:
        rows,vectors,truth,_=load_split(a.root,'ro',split)
        for row,vector,gold in zip(rows,vectors,truth):
            if row['case_id'] in reference:raise ValueError('Duplicate reference')
            reference[row['case_id']]=(gold,probability(selection['model'],vector))
    data=a.root.parent/'v8-stabilization/instruction-sft-01/screen.jsonl'
    rows=[r for r in read(data) if r['language']=='ro'];results=[];changes=0;errors=[]
    for row in rows:
        key=row['case_id'];gold,score=reference[key];n,f,i=native[key],semantic[key],interpreter[key]
        if any(v['gold']!=gold for v in [n,f,i]):raise ValueError('Labels differ')
        value=model.predict(dict(language='ro',turns=row['turns']),n,f,i,focus=row['focus'])
        expected=None if score is None else int(score>=selection['threshold'])
        changes+=value['predicted']!=expected
        if score is None or value['score'] is None:
            if score is not value['score']:raise ValueError('Abstention differs')
        else:errors.append(abs(score-value['score']))
        results.append(dict(case_id=key,gold=gold,kind='conversational' if row.get('source')=='v7_training_frame_projection' else 'formal',**value))
    if len(results)!=200:raise ValueError('Incomplete Romanian screen')
    error=max(errors,default=0.)
    report=dict(format='roguard-installed-three-expert-replay-v1',language='ro',rows=len(results),decision_changes=changes,
        maximum_score_error=error,parity_pass=changes==0 and error<=.001,score_tolerance=.001,
        artifact_sha256=sha(artifact),runtime_source_sha256=sha(runtime.__file__),wheel_sha256=sha(a.wheel),
        component_report_sha256=report_hashes,actual_installed_neural_outputs_used=True,neural_models_loaded_in_this_check=False,
        screen_counts={kind:counts([r for r in results if r['kind']==kind]) for kind in ['formal','conversational']},
        reserved_probes_used=False,stable_release_ready=False,independent_accuracy_evaluation=False)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'predictions.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in results))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if not report['parity_pass']:raise SystemExit('Installed three-expert parity failed')


if __name__=='__main__':main()
