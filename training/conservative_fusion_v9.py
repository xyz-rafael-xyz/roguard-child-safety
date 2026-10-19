"""Development-only full-coverage threshold repair for frozen fusion models."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
from training.review_fusion_v9 import sha, read, probability, metrics

PARENTS = {36: ('review-fusion-36', 'training.review_fusion_v9'),
           76: ('three-expert-fusion-76', 'training.three_expert_fusion_v9')}


def full_coverage_gap(truth, scores):
    if len(truth) != len(scores) or not truth or any(type(y) is not int or y not in (0, 1) for y in truth):
        raise ValueError('Invalid development population')
    if any(s is None or type(s) not in (float, int) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Full coverage requires finite decided development scores')
    positives = [s for y,s in zip(truth,scores) if y]
    if not positives:
        raise ValueError('No development concerns')
    boundary = min(positives)
    below = [s for s in scores if s < boundary]
    if not below:
        return max(0., boundary/2)
    lower = max(below)
    midpoint = lower + (boundary-lower)/2
    return midpoint if lower < midpoint < boundary else boundary


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','audit'))
    p.add_argument('--parent',type=int,choices=PARENTS,required=True)
    p.add_argument('--language',choices=('ro','uk'),required=True);a=p.parse_args()
    root=Path('research/v9-stabilization');study=root/'conservative-fusion-77'
    directory,module=PARENTS[a.parent];source=root/directory/a.language
    output=study/str(a.parent)/a.language
    load=importlib.import_module(module).load_split
    parent_path=source/'selection.json';parent=json.loads(parent_path.read_text())
    rows,x,y,hashes=load(root,a.language,'dev')
    actual=[probability(parent['model'],v) for v in x]
    oof=read(source/'oof-predictions.jsonl')
    if [r['case_id'] for r in rows]!=[r['case_id'] for r in oof] or y!=[r['gold'] for r in oof]:
        raise ValueError('OOF input binding differs')
    original=[r['score'] for r in oof]
    if a.stage=='prepare':
        cutoff=min(parent['selected']['threshold'],full_coverage_gap(y+y,original+actual))
        output.mkdir(parents=True,exist_ok=False)
        record=dict(created_utc=datetime.now(timezone.utc).isoformat(),parent=a.parent,language=a.language,
            parent_selection_sha256=sha(parent_path),protocol_sha256=sha(study/'protocol.json'),source_sha256=sha(__file__),
            development_hashes=hashes,oof_sha256=sha(source/'oof-predictions.jsonl'),threshold=cutoff,
            previous_threshold=parent['selected']['threshold'],model=parent['model'],
            development=dict(oof=metrics(rows,y,original,cutoff),refit=metrics(rows,y,actual,cutoff)),
            calibration_used_for_threshold=False,calibration_diagnosis_informed_experiment=True,
            reserved_probes_used=False,stable_release_ready=False)
        (output/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
    else:
        path=output/'selection.json';record=json.loads(path.read_text())
        if (record['parent_selection_sha256']!=sha(parent_path) or record['source_sha256']!=sha(__file__)
                or record['protocol_sha256']!=sha(study/'protocol.json') or record['development_hashes']!=hashes
                or record['oof_sha256']!=sha(source/'oof-predictions.jsonl')):
            raise ValueError('Frozen conservative threshold evidence changed')
        rows,x,y,hashes=load(root,a.language,'cal')
        scores=[probability(record['model'],v) for v in x]
        counts=metrics(rows,y,scores,record['threshold'])
        passed=all(c['passes'] for groups in record['development'].values() for c in groups.values()) and all(c['passes'] for c in counts.values())
        (output/'cal-predictions.jsonl').write_text(''.join(json.dumps(dict(case_id=r['case_id'],gold=t,score=s,predicted=None if s is None else int(s>=record['threshold'])))+'\n' for r,t,s in zip(rows,y,scores)))
        report=dict(selection_sha256=sha(path),calibration_hashes=hashes,calibration=counts,counts_pass=passed,
            reserved_probes_used=False,stable_release_ready=False,requires_installed_parity_and_reserved_checks=True)
        (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
