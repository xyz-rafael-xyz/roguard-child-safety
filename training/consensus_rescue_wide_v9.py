"""Development-calibrated review rescue from agreeing frozen models."""
import argparse
import gzip
import json
from pathlib import Path
from roguard.evidence import ONTOLOGY, disposition
from training.review_fusion_v9 import read,sha,metrics
from training.reserved_metrics_v9 import counts

ROOT=Path('research/v9-stabilization');STUDY=ROOT/'consensus-rescue-112'


def confidence(native, interpreted, native_threshold):
    if native['score'] is None or native['score'] < native_threshold or interpreted['valid_label_tokens'] is not True:
        return None
    labels=interpreted['interpretation']
    if disposition({k:{'label':v} for k,v in labels.items()})[0]=='CONTINUE':return None
    p=interpreted['conditional_field_probabilities']
    def value(field,label):return p[field][ONTOLOGY[field].index(label)]
    # Alternative real subjects and active assertions support the same review
    # action. This score is a heuristic, not a joint/calibrated probability.
    return min(value('concern','1'),1-value('subject','FICTIONAL'),
               1-value('source','FICTION'),1-value('assertion','NEGATED'))


def rescue_score(base_score, base_threshold, rescue_confidence, rescue_threshold):
    if base_score is None:return None
    margin=base_score-base_threshold
    if rescue_confidence is not None:margin=max(margin,rescue_confidence-rescue_threshold)
    return .5+.5*margin


def load_original(split):
    paths=dict(rows=ROOT/'bare-distress-95/uk'/(split+'.jsonl.gz'),
        native=ROOT/'bare-distress-95/uk'/(split+'-predictions.jsonl'),
        interpreter=ROOT/'interpreter-full-21'/split/'predictions.jsonl',
        base=ROOT/'repaired-three-expert-97/95'/(split+'-predictions.jsonl'))
    rows=read(paths['rows']);parts={k:{r['case_id']:r for r in read(p)} for k,p in paths.items() if k!='rows'}
    if any(len(parts[k])!=len(rows) for k in ('native','base')):raise ValueError('Incomplete sourcecoverage')
    ordered={k:[v[r['case_id']] for r in rows] for k,v in parts.items()}
    truth=[r['gold'] for r in ordered['base']]
    if any(r['gold']!=y or r['language']!='uk' for v in ordered.values() for r,y in zip(v,truth)):
        raise ValueError('Gold/language mismatch')
    return rows,ordered,truth,{k:sha(p) for k,p in paths.items()}


def save(path,rows,truth,scores):
    path.write_text(''.join(json.dumps(dict(case_id=r['case_id'],language='uk',gold=y,score=s,
        predicted=None if s is None else int(s>=.5)))+'\n' for r,y,s in zip(rows,truth,scores)))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('fit','audit'));a=p.parse_args()
    for name,expected in json.loads((STUDY/'source-freeze.json').read_text()).items():
        if sha(name)!=expected:raise ValueError('Frozen recipe differs: '+name)
    parent=ROOT/'repaired-three-expert-97/95/selection.json';base=json.loads(parent.read_text())
    native_path=ROOT/'bare-distress-95/uk/review.json.gz'
    native_threshold=json.loads(gzip.decompress(native_path.read_bytes()))['thresholds']['uk']
    if a.stage=='fit':
        rows,parts,truth,hashes=load_original('dev')
        oof_path=ROOT/'repaired-three-expert-97/95/oof-predictions.jsonl';oof=read(oof_path)
        if [r['case_id'] for r in oof]!=[r['case_id'] for r in rows]:raise ValueError('OOF order differs')
        strengths=[confidence(n,i,native_threshold) for n,i in zip(parts['native'],parts['interpreter'])]
        boundaries=sorted({.5,1.000001}|{s for s in strengths if s is not None and s>=.5})
        thresholds=[.5]+[(x+y)/2 for x,y in zip(boundaries,boundaries[1:])]+[1.000001]
        selected=None
        for threshold in thresholds:
            outputs={name:[rescue_score(v['score'],base['threshold'],s,threshold) for v,s in zip(values,strengths)]
                     for name,values in [('dev',parts['base']),('oof',oof)]}
            groups={name:metrics(rows,truth,scores,.5) for name,scores in outputs.items()}
            if all(g['passes'] and g['false_reviews']*100<=g['negative'] for v in groups.values() for g in v.values()):
                selected=(threshold,outputs,groups);break
        if selected is None:raise ValueError('No baseline-preserving threshold passes1%developmentfalse-reviewlimit')
        threshold,outputs,groups=selected
        record=dict(parent_selection_sha256=sha(parent),native_artifact_sha256=sha(native_path),
            native_threshold=native_threshold,base_threshold=base['threshold'],rescue_threshold=threshold,
            threshold=.5,development=groups['dev'],oof=groups['oof'],development_hashes=dict(hashes,oof=sha(oof_path)),
            policy='Preserve every base review; rescueonlynative-positiveandinterpreter-review withminimumconcern/nonfiction-subject/nonfiction-source/nonnegated-assertionmass abovefixedthreshold.',
            selection='Smallest confidencecutoff>=0.5with <=1%false reviews ineach originalDEVandbase-OOFgroup. StressandCALnotusedforcutoff.',
            score_is_calibrated=False,calibration_used_for_selection=False,stress_used_for_selection=False,
            synthetic_only=True,independent_accuracy_evaluation=False,reserved_probes_used=False,stable_release_ready=False)
        (STUDY/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
        for name,scores in outputs.items():save(STUDY/(name+'-predictions.jsonl'),rows,truth,scores)
        print(json.dumps(record),flush=True)
    else:
        selected=STUDY/'selection.json';r=json.loads(selected.read_text())
        if sha(parent)!=r['parent_selection_sha256'] or sha(native_path)!=r['native_artifact_sha256']:
            raise ValueError('Frozen parent differs')
        rows=read(ROOT/'installed-repaired-99/stress.jsonl')
        base_predictions=read(ROOT/'installed-repaired-99/actual/stress-predictions.jsonl')
        native=read(ROOT/'stress-components-106/actual/native.jsonl');interpreted=read(ROOT/'stress-components-106/actual/interpreter.jsonl')
        ids=[v['case_id'] for v in rows]
        if any([v['case_id'] for v in values]!=ids for values in (base_predictions,native,interpreted)):raise ValueError('Stress order differs')
        truth=[v['gold'] for v in base_predictions]
        scores=[rescue_score(b['score'],r['base_threshold'],confidence(n,i,r['native_threshold']),r['rescue_threshold']) for b,n,i in zip(base_predictions,native,interpreted)]
        stress=counts([dict(gold=y,predicted=None if s is None else int(s>=.5)) for y,s in zip(truth,scores)])
        save(STUDY/'stress-predictions.jsonl',rows,truth,scores)
        report=dict(selection_sha256=sha(selected),stress=stress,reserved_probes_used=False,stable_release_ready=False)
        if stress['passes']:
            rows,parts,truth,hashes=load_original('cal')
            scores=[rescue_score(b['score'],r['base_threshold'],confidence(n,i,r['native_threshold']),r['rescue_threshold']) for b,n,i in zip(parts['base'],parts['native'],parts['interpreter'])]
            groups=metrics(rows,truth,scores,.5)
            report.update(calibration=groups,calibration_hashes=hashes,counts_pass=all(v['passes'] for v in groups.values()),calibration_audit_skipped=False)
            save(STUDY/'cal-predictions.jsonl',rows,truth,scores)
        else:report.update(counts_pass=False,calibration_audit_skipped=True)
        (STUDY/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
