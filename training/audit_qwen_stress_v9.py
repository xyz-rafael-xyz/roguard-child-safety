"""Check the previously failed Ukrainian stress set without selecting on it."""
import json
from pathlib import Path
from training.review_fusion_v9 import read,sha,probability
from training.native_qwen_fusion_v9 import features
from training.reserved_metrics_v9 import counts
from training.prepare_instruction_sft_v8 import target

ROOT=Path('research/v9-stabilization')
STUDY=ROOT/'qwen-stress-93'


def main():
    for name,digest in json.loads((STUDY/'source-freeze.json').read_text()).items():
        if sha(name)!=digest:raise ValueError('Frozen stress recipe differs')
    rows=read(STUDY/'inputs.jsonl')
    qwen=read(STUDY/'inference/predictions.jsonl')
    native=read(ROOT/'needs-repair-85/uk/stress-predictions.jsonl')
    fixed=json.loads((ROOT/'qwen-qualification-90/audit.json').read_text())['thresholds']['uk']
    fitted=json.loads((ROOT/'native-qwen-fusion-92/uk/selection.json').read_text())
    if not len(rows)==len(qwen)==len(native)==216:raise ValueError('Stress coverage differs')
    outputs={'qwen':[],'combined':[]}
    for row,q,n in zip(rows,qwen,native):
        if len({v['case_id'] for v in (row,q,n)})!=1 or q['gold']!=target(row) or n['gold']!=target(row):
            raise ValueError('Stress identity or truth differs')
        vector=features(n,q);score=probability(fitted['model'],vector)
        outputs['qwen'].append(dict(case_id=row['case_id'],gold=target(row),
            predicted=int(q['score']>=fixed) if q['valid_label_token'] else None))
        outputs['combined'].append(dict(case_id=row['case_id'],gold=target(row),
            predicted=int(score>=fitted['threshold']) if score is not None else None))
    report=dict(metrics={name:counts(values) for name,values in outputs.items()},
        selection_used_stress=False,reserved_probes_used=False,stable_release_ready=False,
        data_sha256=sha(STUDY/'inputs.jsonl'),qwen_prediction_sha256=sha(STUDY/'inference/predictions.jsonl'),
        fusion_selection_sha256=sha(ROOT/'native-qwen-fusion-92/uk/selection.json'))
    (STUDY/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
