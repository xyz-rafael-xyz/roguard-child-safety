"""Convert the exposed 5,022-case v9 prior audit without altering its labels."""
import argparse
import gzip
import json
from pathlib import Path

from roguard.trajectory_data import FORMAT, file_sha, prefix, validate_case, write_jsonl
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--selector',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite broad regression')
    a.output.mkdir(parents=True)
    rows=[json.loads(line) for line in gzip.open(a.source/'inputs.jsonl.gz','rt',encoding='utf-8')]
    if len(rows)!=5022:raise ValueError('Incomplete v9 prior population')
    model=TrajectoryModel(a.selector);cases=[];changed=[];missing=0
    for entry in rows:
        r=entry['row'];point={'focus':r['focus'],'review':entry['gold']}
        if 'attributes' in r:point['world']=r['attributes']
        else:missing+=1
        c={'format':FORMAT,'case_id':'broad-'+r['language']+'-'+entry['group']+'-'+r['case_id'],
           'language':r['language'],'turns':r['turns'][:r['focus']+1],
           'split':'challenge','family':'v9-prior-'+entry['group'],
           'author':'project-v9','generator':'v9-prior-'+entry['group'],'synthetic':True,
           'checkpoints':[point],'original_case_id':r['case_id'],'original_group':entry['group']}
        validate_case(c);cases.append(c)
        payload=prefix(c,point['focus'])
        view,inference=prepare_memory(payload,model,1200,preserve_complete=True)
        if inference!=payload or view['mandatory_overflow']:changed.append(c['case_id'])
    write_jsonl(a.output/'cases.jsonl',cases)
    report={'cases':len(cases),'missing_joint_annotations':missing,'changed_inference_inputs':len(changed),
            'changed_case_ids':changed,'selector_sha256':model.identity,
            'source_sha256':file_sha(a.source/'inputs.jsonl.gz'),
            'cases_sha256':file_sha(a.output/'cases.jsonl'),
            'family_note':'Conservatively grouped by original population; no independent-family precision claim.',
            'purpose':'Exposed regression only. No fitting or independent evaluation.',
            'stable_release_ready':False}
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='changed_case_ids'},indent=2))


if __name__=='__main__':main()
