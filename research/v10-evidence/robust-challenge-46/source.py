"""Frozen standalone readout on fresh44, sharing only verified encoder compute."""
import argparse
import gc
import json
from pathlib import Path
import time

from roguard.committee_bundle import load_committee
from roguard.gemma4_committee import Gemma4ReviewCommittee
from roguard.gemma4_review import Gemma4ReviewModel
from roguard.trajectory_data import file_sha, input_identity, read_jsonl, write_jsonl
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','bundle','selector','selected_study','selected','data','reference','study'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    a=p.parse_args()
    if a.study.exists():raise ValueError('Refusing to overwrite readout diagnostic')
    selected=json.loads((a.selected_study/'report.json').read_text())
    reference=json.loads((a.reference/'report.json').read_text());prior=json.loads((a.reference/'protocol.json').read_text())
    integration=json.loads(Path('research/v10-evidence/committee-trajectory-42/protocol.json').read_text())
    if (not selected['complete'] or not reference['complete']
            or file_sha(a.reference/'predictions.jsonl')!=reference['predictions_sha256']
            or file_sha(a.data/'cases.jsonl')!=prior['cases_sha256']
            or file_sha(a.selector)!=integration['selector_sha256']):
        raise ValueError('Frozen selection, data or baseline binding differs')
    baseline={r['case_id']:r for r in read_jsonl(a.reference/'predictions.jsonl')}
    rows=read_jsonl(a.data/'cases.jsonl');selector=TrajectoryModel(a.selector,expected_sha256=integration['selector_sha256'])
    if len(rows)!=72 or set(baseline)!={r['case_id'] for r in rows}:raise ValueError('Fresh44 checkpoint mismatch')
    a.study.mkdir(parents=True)
    protocol={'format':'roguard-readout-fresh44-diagnostic-1','source_sha256':file_sha(__file__),
        'selection_report_sha256':file_sha(a.selected_study/'report.json'),
        'reference_report_sha256':file_sha(a.reference/'report.json'),'cases_sha256':file_sha(a.data/'cases.jsonl'),
        'bundle_sha256':integration['bundle_sha256'],'selector_sha256':integration['selector_sha256'],
        'amendment_sha256':file_sha('research/v10-evidence/protocol-amendment-44.json'),
        'decision_rule':'standalone selected readout, frozen original-DEV threshold; no voting combination',
        'fitting_performed':False,'independent_validation':False,'stable_release_ready':False,'languages':{}}
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    records=[];groups={};started=time.monotonic()
    for lang in ('ro','uk'):
        # Validate every original bundle asset before using its v9/attention files.
        load_committee(a.bundle,expected_sha256=integration['bundle_sha256'],base=a.base,language=lang)
        manifest=json.loads(a.bundle.read_text());parts={}
        for kind in ('v9','attention'):
            entry=manifest['languages'][lang][kind]
            parts[kind]=Gemma4ReviewModel(a.bundle.parent/entry['artifact'],a.bundle.parent/entry['adapter'],a.base,expected_sha256=entry['artifact_sha256'])
        spec=a.selected_study/(lang+'-review.json');adapter=a.selected/(lang+'-selected.safetensors')
        if file_sha(adapter)!=selected['languages'][lang]['selected']['adapter_sha256']:
            raise ValueError('Selected adapter differs from frozen report')
        parts['readout']=Gemma4ReviewModel(spec,adapter,a.base,expected_sha256=file_sha(spec))
        model=Gemma4ReviewCommittee(**parts)
        protocol['languages'][lang]={'readout_spec_sha256':file_sha(spec),'readout_identity':parts['readout'].identity,
            'v9_identity':parts['v9'].identity,'compute_engine_identity':model.identity}
        (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        batch=[];selected_rows=[]
        for row in (r for r in rows if r['language']==lang):
            payload={'language':lang,'turns':row['turns']};view,inference=prepare_memory(payload,selector,1200,preserve_complete=True)
            old=baseline[row['case_id']]['prediction']
            if (view['mandatory_overflow'] or len(inference['turns'])>12
                    or old['input_digest']!=input_identity(payload) or old['inference_input_digest']!=input_identity(inference)):
                raise ValueError('Fresh44 selected input differs')
            batch.append(inference);selected_rows.append(row)
        results=model.predict_many(batch)
        for row,result in zip(selected_rows,results):
            old=baseline[row['case_id']]['prediction'];components=result['components']
            if components is None or components['v9']!=old['components']['v9']:
                raise ValueError('Fresh frozen-v9 score/logit parity failed')
            gold=row['checkpoints'][0]['review'];candidate=components['readout']['predicted'];v9=components['v9']['predicted']
            records.append({'case_id':row['case_id'],'gold':gold,'input_digest':old['input_digest'],
                'inference_input_digest':old['inference_input_digest'],'candidate':components['readout'],'v9':components['v9'],
                'candidate_identity':parts['readout'].identity,'qualified':False})
            g=groups.setdefault(lang+':'+row['context_kind'],{'rows':0,'candidate_errors':0,'v9_errors':0,'fixed':0,'introduced':0})
            g['rows']+=1;g['candidate_errors']+=candidate!=gold;g['v9_errors']+=v9!=gold
            g['fixed']+=candidate==gold and v9!=gold;g['introduced']+=candidate!=gold and v9==gold
        write_jsonl(a.study/'predictions.jsonl',records)
        del model,parts;gc.collect()
        print(json.dumps({'language':lang,'done':len(records),'groups':groups}),flush=True)
    totals={k:sum(g[k] for g in groups.values()) for k in next(iter(groups.values()))}
    report={'complete':True,'groups':groups,'totals':totals,'seconds':time.monotonic()-started,
        'fresh_v9_components_exact':True,'predictions_sha256':file_sha(a.study/'predictions.jsonl'),
        'decision':'REJECT' if totals['candidate_errors'] else 'BINARY_DIAGNOSTIC_PASS_ONLY',
        'independent_validation':False,'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
