"""Measure the actual evidence runtime's focal-turn review decision."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from roguard.evidence import load_evidence_model, analyze_evidence
from roguard.review_evaluation import evaluate
from training.prepare_instruction_sft_v8 import target


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=('character','semantic','guarded'),default='character')
    p.add_argument('--base');p.add_argument('--device',default='cpu');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(l) for l in a.data.read_text().splitlines()]
    model=load_evidence_model(a.backend,a.base,a.device)
    expected=[];predictions=[];start=time.monotonic()
    for r in rows:
        tick=time.monotonic()
        report=analyze_evidence({'language':r['language'],'turns':r['turns'][:r['focus']+1]},model)
        focal=[e for e in report['evidence'] if e['reference']['turn_index']==r['focus']]
        if not focal:raise ValueError('Missing focal evidence')
        predictions.append(dict(case_id=r['case_id'],language=r['language'],predicted=int(any(e['proposed_action']!='CONTINUE' for e in focal)),
            seconds=time.monotonic()-tick,interpretations=[e['interpretation_summary'] for e in focal],
            references=[e['reference'] for e in focal],external_action_taken=report['external_action_taken']))
        expected.append(dict(case_id=r['case_id'],language=r['language'],gold=target(r),
            kind='conversational' if r.get('source')=='v7_training_frame_projection' else 'formal',
            invariant_group=r.get('invariant_group'),pair_group=r.get('pair_group')))
    result=evaluate(expected,predictions)
    result.update(seconds=time.monotonic()-start,backend=a.backend,model_identity=model.identity,
                  input_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
                  task='focal-turn review request; unresolved attribute proposals can request review')
    for name,data in [('expected',expected),('predictions',predictions)]:
        (a.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
