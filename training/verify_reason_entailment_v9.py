"""Verify actual installed separate-cutoff support/revision entailment weights, then conditionally open frozen test."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from training.train_bare_distress_v9 import digest,read
from training.prepare_instruction_sft_v8 import target
from training.verify_candidate_pipeline_v9 import counts


def qualified(record):
    if record.get('complete') is not True or record.get('counts_pass') is not True or record.get('fresh_test_not_inferred') is not True:return False
    selected=[v for v in record.get('epochs',[]) if v['epoch']==record.get('selected_epoch')]
    if len(selected)!=1 or set(selected[0]['metrics'])!={'ro_dev','uk_dev','ro_scope_dev','uk_scope_dev','uk_stress'} or set(record.get('audit',{}))!={'ro_cal','uk_cal','ro_exposed','uk_exposed'}:return False
    for split,groups in list(selected[0]['metrics'].items())+list(record['audit'].items()):
        expected={'formal','conversational'} if split in ('ro_dev','uk_dev','ro_cal','uk_cal') else {'all'}
        if set(groups)!=expected:return False
        for g in groups.values():
            values=[g.get(k) for k in ('positive','negative','missed','false_reviews','abstentions')]
            if any(type(v) is not int for v in values):return False
            pos,neg,missed,false,abstain=values
            if pos<=0 or neg<=0 or not 0<=missed<=pos or not 0<=false<=neg or abstain or missed*100>pos or false*20>neg:return False
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--installed-root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--wheel',type=Path,required=True);a=p.parse_args()
    study=Path('research/v9-stabilization/reason-entailment-installed-124');parent=Path('research/v9-stabilization/reason-coverage-123/candidate')
    import roguard.reason_entailment as runtime
    import torch
    torch.set_num_threads(4)
    if a.installed_root.resolve() not in Path(runtime.__file__).resolve().parents:raise ValueError('Not isolated installed runtime')
    for path,expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(Path(path))!=expected:raise ValueError('Frozen source changed: '+path)
        if path.startswith('src/roguard/') and digest(a.installed_root/path.removeprefix('src/'))!=expected:raise ValueError('Installed source differs')
    record=json.loads((parent/'development.json').read_text());spec=json.loads((parent/'review.json').read_text());ready=qualified(record)
    if record.get('artifact_sha256')!=digest(parent/'review.json') or spec['checkpoint_sha256']!=digest(parent/'selected-model.safetensors'):raise ValueError('Artifact identity differs')
    output=study/'actual';output.mkdir(exist_ok=False)
    (output/'selection.json').write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),artifact_sha256=digest(parent/'review.json'),report_sha256=digest(parent/'development.json'),wheel_sha256=digest(a.wheel),qualified_before_replay=ready),indent=2)+'\n')
    model=runtime.ReasonEntailmentReview(parent/'review.json',parent/'selected-model.safetensors',a.base,expected_sha256=digest(parent/'review.json'),device='mps')
    results={};cpu_rows=[];cpu_refs=[]
    for split in ('ro_dev','uk_dev','ro_scope_dev','uk_scope_dev','uk_stress','ro_cal','uk_cal','ro_exposed','uk_exposed'):
        rows=read(parent/(split+'.jsonl.gz'));reference=[json.loads(l) for l in (parent/(split+'-predictions.jsonl')).read_text().splitlines()]
        if [r['case_id'] for r in rows]!=[r['case_id'] for r in reference]:raise ValueError('Reference coverage mismatch')
        if not ready and split in ('ro_dev','uk_dev','ro_cal','uk_cal'):
            indices=[]
            for native in (False,True):
                for gold in (0,1):indices.extend([i for i,r in enumerate(rows) if (r.get('source')=='v7_training_frame_projection')==native and reference[i]['gold']==gold][:25])
            rows=[rows[i] for i in indices];reference=[reference[i] for i in indices]
        values=[];start=time.monotonic()
        for offset in range(0,len(rows),64):
            batch=rows[offset:offset+64]
            outputs=model.predict_many([dict(language=r['language'],turns=r['turns']) for r in batch],focuses=[r.get('focus',len(r['turns'])-1) for r in batch])
            for r,v,ref in zip(batch,outputs,reference[offset:offset+64]):
                gold=int(r['review']) if 'review' in r else target(r)
                if ref['gold']!=gold:raise ValueError('Gold binding changed')
                values.append(dict(v,case_id=r['case_id'],gold=gold))
            print(split,len(values),'/',len(rows),flush=True)
        if len(values)!=len(reference) or any(v['score'] is None for v in values):raise ValueError('Incomplete inference')
        error=max(abs(v[k]-r[k]) for v,r in zip(values,reference) for k in ('score','support_score','revision_score'));changes=sum(v['predicted']!=r['predicted'] for v,r in zip(values,reference))
        results[split]=dict(rows=len(values),maximum_score_error=error,decision_changes=changes,parity_pass=error<=.001 and changes==0,seconds=time.monotonic()-start,counts=counts(values))
        (output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
        if split in ('ro_scope_dev','uk_scope_dev'):cpu_rows+=rows[:20];cpu_refs+=values[:20]
    del model
    import gc
    gc.collect();torch.mps.empty_cache()
    cpu=runtime.ReasonEntailmentReview(parent/'review.json',parent/'selected-model.safetensors',a.base,expected_sha256=digest(parent/'review.json'),device='cpu')
    actual=cpu.predict_many([dict(language=r['language'],turns=r['turns']) for r in cpu_rows],focuses=[r['focus'] for r in cpu_rows])
    error=max(abs(v[k]-r[k]) for v,r in zip(actual,cpu_refs) for k in ('score','support_score','revision_score'));changes=sum(v['predicted']!=r['predicted'] for v,r in zip(actual,cpu_refs));del cpu
    cpu_result=dict(rows=len(actual),maximum_score_error=error,decision_changes=changes,parity_pass=error<=.001 and changes==0)
    parity=all(v['parity_pass'] for v in results.values()) and cpu_result['parity_pass']
    report=dict(complete=True,installed=results,cpu_smoke=cpu_result,parity_pass=parity,prerequisite_accuracy_pass=ready,synthetic_only=True,independently_annotated=False,stable_release_ready=False)
    if not ready or not parity:report.update(fresh_test_inferred=False,fresh_test_status='skipped_prerequisites_failed')
    else:
        testroot=Path('research/v9-stabilization/paired-scope-117/data');earlier=Path('research/v9-stabilization/paired-installed-118/actual/test-access.json')
        if earlier.exists() or any((Path('research/v9-stabilization')/name/'actual/test-access.json').exists() for name in ('paired-nli-installed-120','policy-entailment-installed-122')):raise ValueError('Test previously opened; cannot claim fresh inference')
        (output/'test-access.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=digest(output/'selection.json'),first_neural_inference=True),indent=2)+'\n')
        model=runtime.ReasonEntailmentReview(parent/'review.json',parent/'selected-model.safetensors',a.base,expected_sha256=digest(parent/'review.json'),device='mps');fresh={}
        for lang in ('ro','uk'):
            path=testroot/(lang+'-test.jsonl');rows=[json.loads(l) for l in path.read_text().splitlines()];values=[]
            for offset in range(0,len(rows),64):
                batch=rows[offset:offset+64]
                outputs=model.predict_many([dict(language=lang,turns=r['turns']) for r in batch],focuses=[r['focus'] for r in batch])
                values.extend(dict(v,case_id=r['case_id'],gold=r['review']) for r,v in zip(batch,outputs))
                (output/(lang+'-test-predictions.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
            fresh[lang]=dict(counts=counts(values),input_sha256=digest(path))
        report.update(fresh_test_inferred=True,fresh=fresh,fresh_counts_pass=all(v['counts']['passes'] for v in fresh.values()))
    report['finished_utc']=datetime.now(timezone.utc).isoformat();(output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
