"""Decode six conditional fields with cached local inference and strict scoring."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from training.semantic_interpreter_v9 import FIELDS,aligned_tokens,chat
from training.prepare_instruction_sft_v8 import target
from roguard.evidence import ONTOLOGY,disposition
from training.gemma_projection_v9 import language_model,last_logits


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--adapter',type=Path)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    mx.set_cache_limit(512*1024*1024)
    model,tok=load(a.base,adapter_path=str(a.adapter) if a.adapter else None);model.set_dtype(mx.float32)
    ids=[tok.encode(str(i),add_special_tokens=False) for i in range(4)]
    if any(len(v)!=1 for v in ids):raise ValueError('Class tokens are not single tokens')
    ids=[v[0] for v in ids]
    rows=[json.loads(l) for l in a.data.read_text().splitlines()]
    prepared=[aligned_tokens(tok,chat(row),{f:0 for f in FIELDS}) for row in rows]
    if not rows or max(len(t) for t,p in prepared)>1024:raise ValueError('Empty or overlength inputs')
    common=0
    for values in zip(*(t[:p[0]] for t,p in prepared)):
        if len(set(values))!=1:break
        common+=1
    common=min(common,min(p[0] for t,p in prepared)-1)
    prefix=make_prompt_cache(model)
    if common:mx.eval(language_model(model).model(mx.array([prepared[0][0][:common]]),cache=prefix))
    expected=[dict(case_id=r['case_id'],language=r['language'],gold=target(r),kind='conversational' if r.get('source')=='v7_training_frame_projection' else 'formal') for r in rows]
    predictions=[];field_correct={f:0 for f in FIELDS};start=time.monotonic();max_error=0.
    with (a.output/'predictions.jsonl').open('w') as out:
        for i,(row,(template,positions)) in enumerate(zip(rows,prepared)):
            tokens=list(template);cache=deepcopy(prefix);cursor=common;values={};probabilities={};valid=True
            for field,position in zip(FIELDS,positions):
                logits=last_logits(model,mx.array([tokens[cursor:position]]),cache=cache)[0,-1].astype(mx.float32)
                allowed=ids[:len(ONTOLOGY[field])];probs=mx.softmax(logits[mx.array(allowed)]);top=mx.argmax(logits)
                mx.eval(probs,top)
                if i<2:
                    full=model(mx.array([tokens[:position]]))[0,-1].astype(mx.float32)
                    check=mx.softmax(full[mx.array(allowed)]);mx.eval(check)
                    error=float(mx.max(mx.abs(check-probs)).item());max_error=max(max_error,error)
                    if error>.001 or int(mx.argmax(full).item())!=int(top.item()):raise ValueError('Cached/full inference mismatch')
                choice=int(mx.argmax(probs).item());tokens[position]=allowed[choice];cursor=position
                values[field]=ONTOLOGY[field][choice];probabilities[field]=[float(v) for v in probs.tolist()]
                valid &= int(top.item()) in allowed
                field_correct[field]+=values[field]==row['attributes'][field]
            decision=int(disposition({k:{'label':v} for k,v in values.items()})[0]!='CONTINUE') if valid else None
            result=dict(case_id=row['case_id'],language=row['language'],gold=target(row),predicted=decision,
                interpretation=values,conditional_field_probabilities=probabilities,valid_label_tokens=valid)
            predictions.append(result);out.write(json.dumps(result)+'\n');out.flush()
            if (i+1)%25==0:print(i+1,'correct',sum(p['predicted']==p['gold'] for p in predictions),'invalid',sum(p['predicted'] is None for p in predictions),flush=True)
    from roguard.review_evaluation import evaluate
    record=dict(synthetic_only=True,stable_release_ready=False,base_revision=Path(a.base).name,
        data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),seconds=time.monotonic()-start,
        field_accuracy={k:v/len(rows) for k,v in field_correct.items()},strict_evaluation=evaluate(expected,predictions),
        cached_prefix_tokens=common,cache_check_max_error=max_error,
        conditional_probabilities_are_not_calibrated=True,reserved_probes_used=False)
    (a.output/'expected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected))
    (a.output/'metrics.json').write_text(json.dumps(record,indent=2)+'\n');print('FINISHED',json.dumps(record),flush=True)


if __name__=='__main__':main()
