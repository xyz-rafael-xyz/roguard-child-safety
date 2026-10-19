"""Development-only threshold selection and separate Qwen calibration audit."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from training.prepare_instruction_sft_v8 import target
from training.conservative_fusion_v9 import full_coverage_gap
from roguard.review_evaluation import evaluate, target_failures

ROOT = Path('research/v9-stabilization')
STUDY = ROOT/'qwen-qualification-90'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def population():
    rows = []
    for split in ('dev', 'cal'):
        for language in ('ro', 'uk'):
            path = ROOT/('native-full-22-'+language)/(split+'.jsonl.gz')
            for line in gzip.decompress(path.read_bytes()).splitlines():
                row = json.loads(line)
                row['qualification_split'] = split
                rows.append(row)
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated qualification identities')
    return rows


def audit(rows, predictions):
    if [r['case_id'] for r in rows] != [p['case_id'] for p in predictions]:
        raise ValueError('Incomplete or reordered predictions')
    for p in predictions:
        if type(p["valid_label_token"]) is not bool or type(p["score"]) not in (int,float) or not math.isfinite(p["score"]) or not 0 <= p["score"] <= 1:
            raise ValueError("Invalid score or validity flag")
    thresholds = {}
    for language in ('ro', 'uk'):
        selected = [(r, p) for r,p in zip(rows,predictions)
                    if r['language']==language and r['qualification_split']=='dev']
        thresholds[language] = full_coverage_gap(
            [target(r) for r,p in selected], [p['score'] for r,p in selected])
    reports = {}
    for split in ('dev','cal'):
        expected, actual = [], []
        for r,p in zip(rows,predictions):
            if r['qualification_split'] != split:
                continue
            if p['gold'] != target(r) or p['language'] != r['language']:
                raise ValueError('Prediction truth metadata differs')
            expected.append(dict(case_id=r['case_id'],language=r['language'],
                gold=target(r),kind='conversational' if r.get('source')=='v7_training_frame_projection' else 'formal'))
            actual.append(dict(case_id=r['case_id'],predicted=int(p['score']>=thresholds[r['language']]) if p['valid_label_token'] else None))
        reports[split] = evaluate(expected, actual)
    failures = {split: target_failures(report) for split,report in reports.items()}
    return dict(target_failures=failures,observed_targets_pass=not any(failures.values()),thresholds=thresholds,selection='Development full-coverage midpoint; calibration never used for threshold selection',
                reports=reports,reserved_probes_used=False,stable_release_ready=False,
                synthetic_only=True,independent_accuracy_evaluation=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','audit'))
    args=parser.parse_args()
    if args.stage=='prepare':
        STUDY.mkdir(exist_ok=False)
        rows=population()
        (STUDY/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        print(len(rows))
    else:
        for name,digest in json.loads((STUDY/'source-freeze.json').read_text()).items():
            if sha(name)!=digest:
                raise ValueError('Frozen qualification source differs: '+name)
        rows=[json.loads(s) for s in (STUDY/'inputs.jsonl').read_text().splitlines()]
        predictions=[json.loads(s) for s in (STUDY/'inference/predictions.jsonl').read_text().splitlines()]
        report=audit(rows,predictions)
        report['predictions_sha256']=sha(STUDY/'inference/predictions.jsonl')
        report['inputs_sha256']=sha(STUDY/'inputs.jsonl')
        (STUDY/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__':
    main()
