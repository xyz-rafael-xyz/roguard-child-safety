"""Installed scope-operator native replay, then conditional first-use scope test."""
import argparse
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import time
from training.train_bare_distress_v9 import digest,read
from training.verify_candidate_pipeline_v9 import counts


def require_groups(report):
    if report.get('complete') is not True or report.get('fresh_test_not_inferred') is not True or report.get('counts_pass') is not True:return False
    language=report.get('language')
    if language not in ('ro','uk'):return False
    selected=[v for v in report.get('epochs',[]) if v['epoch']==report.get('selected_epoch')]
    if len(selected)!=1:return False
    expected={'dev','scope_dev'}|({'stress'} if language=='uk' else set())
    if set(selected[0].get('metrics',{}))!=expected or set(report.get('audit',{}))!={'cal','exposed'}:return False
    for split,groups in list(selected[0]['metrics'].items())+list(report['audit'].items()):
        if set(groups)!=({'formal','conversational'} if split in ('dev','cal') else {'all'}):return False
        for g in groups.values():
            values=[g.get(k) for k in ('positive','negative','missed','false_reviews','abstentions')]
            if any(type(v) is not int for v in values):return False
            pos,neg,missed,false,abstain=values
            if pos<=0 or neg<=0 or not 0<=missed<=pos or not 0<=false<=neg or abstain or missed*100>pos or false*20>neg:return False
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--installed-root',type=Path,required=True);p.add_argument('--wheel',type=Path,required=True);p.add_argument('--ro-base',type=Path,required=True);p.add_argument('--uk-base',type=Path,required=True);a=p.parse_args()
    study=Path('research/v9-stabilization/native-scope-operators-installed-132');parent=Path('research/v9-stabilization/native-scope-operators-131')
    import roguard.native_review as runtime
    import torch
    torch.set_num_threads(4)
    if a.installed_root.resolve() not in Path(runtime.__file__).resolve().parents:raise ValueError('Not installed runtime')
    for path,expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(Path(path))!=expected:raise ValueError('Frozen source differs')
        if path.startswith('src/roguard/') and digest(a.installed_root/path.removeprefix('src/'))!=expected:raise ValueError('Installed source differs')
    output=study/'actual';output.mkdir(exist_ok=False)
    reports={lang:json.loads((parent/lang/'development.json').read_text()) for lang in ('ro','uk')}
    evidence={lang:{name:digest(parent/lang/name) for name in ('development.json','review.json.gz','selected-model.safetensors')} for lang in reports}
    (output/'selection.json').write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),candidates=evidence,wheel_sha256=digest(a.wheel),training_source_freeze_sha256=digest(parent/'source-freeze.json')),indent=2)+'\n')
    results={}
    def load(lang):
        directory=parent/lang
        return runtime.NativeReviewModel(directory/'review.json.gz',directory/'selected-model.safetensors',getattr(a,lang+'_base'),Path('research/v9-stabilization/native-full-22-base-manifest.json'),expected_artifact_sha256=evidence[lang]['review.json.gz'],device='cpu')
    for lang in ('ro','uk'):
        model=load(lang);groups={}
        for split in ('dev','scope_dev','cal','exposed')+(('stress',) if lang=='uk' else ()):
            rows=read(parent/lang/(split+'.jsonl.gz'))
            reference=[json.loads(l) for l in (parent/lang/(split+'-predictions.jsonl')).read_text().splitlines()]
            if [r['case_id'] for r in rows]!=[r['case_id'] for r in reference]:raise ValueError('Reference coverage differs')
            values=[];start=time.monotonic()
            for offset in range(0,len(rows),64):
                batch=rows[offset:offset+64]
                pred=model.predict_many([dict(language=lang,turns=r['turns']) for r in batch],focuses=[r.get('focus',len(r['turns'])-1) for r in batch])
                values.extend(dict(v,case_id=r['case_id'],gold=old['gold']) for v,r,old in zip(pred,batch,reference[offset:offset+64]))
            maximum=max(abs(v['score']-r['score']) for v,r in zip(values,reference));changes=sum(v['predicted']!=r['predicted'] for v,r in zip(values,reference))
            if len(values)!=len(rows):raise ValueError('Incomplete replay')
            groups[split]=dict(rows=len(rows),maximum_score_error=maximum,decision_changes=changes,parity_pass=maximum<=.001 and changes==0,seconds=time.monotonic()-start)
            (output/(lang+'-'+split+'-predictions.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
            print(lang,split,json.dumps(groups[split]),flush=True)
        results[lang]=groups;del model
    parity=all(g['parity_pass'] for groups in results.values() for g in groups.values())
    qualification=all(require_groups(r) for r in reports.values())
    report=dict(installed_parity=results,parity_pass=parity,prerequisite_accuracy_pass=qualification,synthetic_only=True,independently_annotated=False,stable_release_ready=False)
    if not parity or not qualification:
        report.update(fresh_test_status='skipped_prerequisites_failed',fresh_test_inferred=False)
    else:
        previous=('paired-installed-118','paired-nli-installed-120','policy-entailment-installed-122','reason-entailment-installed-124','native-coverage-installed-126','native-morphology-installed-129')
        if any((Path('research/v9-stabilization')/name/'actual/test-access.json').exists() for name in previous):raise ValueError('Frozen test already opened; fresh evaluation unavailable')
        (output/'test-access.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=digest(output/'selection.json'),first_neural_inference_for_this_frozen_test=True),indent=2)+'\n')
        fresh={}
        for lang in ('ro','uk'):
            model=load(lang);path=Path('research/v9-stabilization/paired-scope-117/data')/(lang+'-test.jsonl');rows=[json.loads(l) for l in path.read_text().splitlines()];values=[]
            for offset in range(0,len(rows),64):
                batch=rows[offset:offset+64]
                pred=model.predict_many([dict(language=lang,turns=r['turns']) for r in batch],focuses=[r['focus'] for r in batch])
                values.extend(dict(v,case_id=r['case_id'],gold=r['review']) for r,v in zip(batch,pred))
                (output/(lang+'-test-predictions.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
            fresh[lang]=dict(counts=counts(values),input_sha256=digest(path));del model
        report.update(fresh_test_status='completed',fresh_test_inferred=True,fresh=fresh,fresh_counts_pass=all(v['counts']['passes'] for v in fresh.values()))
    report.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat());(output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
