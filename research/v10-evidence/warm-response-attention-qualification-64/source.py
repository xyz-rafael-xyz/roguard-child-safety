"""Read-only completed-study audit; saved evidence verification, not new inference."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

LABELS=('appropriate','boundary','privacy','pressure','unsupported')
def check(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(s) for s in path.read_text().splitlines()]
def verify(root,private,name,protocol_sha):
    evidence=root/'research/v10-evidence';public=evidence/name;weights=private/name
    report=read(public/'report.json');selection=read(public/'selection.json');protocol=read(public/'protocol.json');model=read(public/'model.json');progress=read(public/'progress.json')
    check(report['complete'] is True and selection['complete'] is True,'Incomplete study')
    check(sha(public/'protocol.json')==protocol_sha==model['protocol_sha256'],'Protocol mismatch')
    check(sha(public/'source.py')==protocol['source_sha256'],'Source snapshot mismatch')
    check(sha(public/'helper.py')==protocol['helper_sha256'],'Helper snapshot mismatch')
    for path,digest in protocol['helpers'].items():check(sha(root/path)==digest,'Shared helper mismatch: '+path)
    check(sha(public/'selection.json')==report['selection_sha256']==model['selection_sha256'],'Selection mismatch')
    check(sha(public/'model.json')==report['model_spec_sha256'],'Model specification mismatch')
    check(report['selected']==selection['selected']==progress['selected'],'Selected metadata differs')
    check(selection['epochs']==progress['epochs'],'History differs')
    epochs=selection['epochs'];check([e['epoch'] for e in epochs]==list(range(protocol['optimization']['epochs']+1)),'Missing epoch')
    for epoch in epochs:
        check(sha(weights/f"epoch-{epoch['epoch']}.safetensors")==epoch['adapter_sha256'],'Checkpoint differs')
        dev=epoch['development'];populations=dev['populations'];check(set(populations)=={'historical','generated'},'Missing DEV population')
        worst=[]
        for origin,p in populations.items():
            expected={f'{lang}:{gold}':(8 if origin=='historical' else 128 if gold=='appropriate' else 32) for lang in ('ro','uk') for gold in LABELS}
            check(set(p['groups'])==set(expected),'Missing DEV category')
            for key,g in p['groups'].items():check(g['total']==expected[key] and type(g['errors']) is int and 0<=g['errors']<=g['total'],'Invalid group count')
            check(p['rows']==sum(expected.values()) and p['errors']==sum(g['errors'] for g in p['groups'].values()) and p['correct']==p['rows']-p['errors'],'Invalid population count')
            check(math.isfinite(p['cross_entropy']) and p['cross_entropy']>=0,'Invalid population loss')
            rate=max(g['errors']/g['total'] for g in p['groups'].values());worst.append(rate)
            check(p['selection_rank']==[rate,p['errors'],p['cross_entropy']],'Invalid population rank')
        rank=[max(worst),max(p['errors']/p['rows'] for p in populations.values()),sum(p['errors'] for p in populations.values()),sum(p['cross_entropy'] for p in populations.values())/2]
        check(dev['selection_rank']==rank and dev['errors']==rank[2],'Invalid combined rank')
    chosen=min(epochs,key=lambda e:e['development']['selection_rank'])
    check(chosen==selection['selected'],'Wrong registered selection')
    check(sha(weights/'selected.safetensors')==chosen['adapter_sha256']==model['adapter_sha256'],'Selected weight mismatch')
    data_report=evidence/'response-curriculum-25/report.json';generated_report=evidence/'compositional-response-54/report.json';context_protocol=evidence/'response-context-34/protocol.json'
    for path,key in [(data_report,'data_report_sha256'),(generated_report,'generated_report_sha256'),(context_protocol,'context_protocol_sha256')]:check(sha(path)==protocol['bindings'][key],'Gold-source report changed')
    data,generated,context=read(data_report),read(generated_report),read(context_protocol)
    sources={'historical-development':(evidence/'response-curriculum-25/dev.jsonl',data['artifacts']['dev'],80),
             'generated-development':(private/'compositional-response-54/dev.jsonl',generated['artifacts']['dev'],512),
             'context34':(evidence/'response-context-34/inputs.jsonl',context['input_sha256'],120),
             'old-diagnostic':(evidence/'response-curriculum-25/old-diagnostic.jsonl',data['artifacts']['old-diagnostic'],10)}
    check(set(report['metrics'])==set(sources),'Wrong final evaluation populations')
    paired={}
    for population,(source,digest,size) in sources.items():
        check(sha(source)==digest,'Gold inputs changed')
        gold=lines(source);records=lines(weights/(population+'-predictions.jsonl'))
        check(len(gold)==len(records)==size and len({r['case_id'] for r in records})==size,'Wrong prediction population')
        expected={r['case_id']:r for r in gold};check(set(expected)=={r['case_id'] for r in records},'Prediction identity mismatch')
        metrics=report['metrics'][population]
        for row in records:
            check(all(row.get(k)==v for k,v in expected[row['case_id']].items()),'Prediction input or gold differs')
        for key,summary_key in [('prediction','candidate'),('baseline26','baseline26')]:
            groups={f'{lang}:{gold}':{'errors':0,'total':0} for lang in ('ro','uk') for gold in LABELS};losses=[]
            for row in records:
                pred=row[key];p=pred['probabilities']
                check(set(p)==set(LABELS) and all(math.isfinite(x) and 0<=x<=1 for x in p.values()) and math.isclose(sum(p.values()),1,abs_tol=1e-9),'Invalid probabilities')
                check(pred['label']==max(LABELS,key=lambda label:p[label]),'Label does not match probabilities')
                group=groups[row['language']+':'+row['gold']];group['total']+=1;group['errors']+=pred['label']!=row['gold']
                losses.append(-math.log(max(p[row['gold']],1e-30)))
            summary=metrics[summary_key];errors=sum(g['errors'] for g in groups.values())
            check(summary['groups']==groups and summary['errors']==errors and summary['correct']==size-errors and summary['rows']==size,'Saved prediction counts differ')
            check(summary['unique_drafts']==len({(r['language'],r['draft']) for r in records}),'Draft count differs')
            check(math.isclose(summary['cross_entropy'],sum(losses)/size,rel_tol=1e-12,abs_tol=1e-12),'Saved prediction loss differs')
        fixed=sum(r['baseline26']['label']!=r['gold'] and r['prediction']['label']==r['gold'] for r in records)
        introduced=sum(r['baseline26']['label']==r['gold'] and r['prediction']['label']!=r['gold'] for r in records)
        check((fixed,introduced)==(metrics['fixed'],metrics['introduced']),'Paired counts differ')
        if population.endswith('-development'):
            origin=population.split('-')[0];check(metrics['candidate']==chosen['development']['populations'][origin],'Selected DEV metrics differ')
        paired[population]={'rows':size,'errors':metrics['candidate']['errors'],'baseline_errors':metrics['baseline26']['errors'],'fixed':fixed,'introduced':introduced,'predictions_sha256':sha(weights/(population+'-predictions.jsonl'))}
    for key,limit in [('maximum_initial_feature_error',1e-3),('maximum_full_replay_error',1e-3),('selected_reload_maximum_logit_error',1e-5)]:check(math.isfinite(report[key]) and 0<=report[key]<=limit,'Reported runtime check failed')
    rejected=any(p['errors'] or p['introduced'] for p in paired.values())
    decision='REJECT' if rejected else 'RESPONSE_DEVELOPMENT_PASS_REQUIRES_RESERVED_TEST'
    check(report['decision']==decision,'Decision differs from observations')
    check(report['training_rows']==12884 and report['development_rows']==592 and report['trainable_tensors']==74,'Training inventory differs')
    check(report['held_out_test_opened'] is False and report['stable_release_ready'] is False and report['independent_validation'] is False,'Unsupported qualification claim')
    return {'verification':'PASS','scope':'Completed saved artifacts, source bindings, registered selection and paired prediction counts/losses only; no new model inference.',
            'study':name,'report_sha256':sha(public/'report.json'),'protocol_sha256':protocol_sha,'selection_sha256':sha(public/'selection.json'),
            'selected_epoch':chosen['epoch'],'checkpoints_verified':len(epochs),'paired_predictions':paired,'decision':decision,'model_qualified':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--study',required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite verification')
    result=verify(a.root,a.private,a.study,a.protocol_sha256);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
