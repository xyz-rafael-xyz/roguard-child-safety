"""Decode equal-layout groups with an independently repeated common prefix cache."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from training.semantic_interpreter_v9 import FIELDS, aligned_tokens, chat
from training.prepare_instruction_sft_v8 import target
from training.gemma_projection_v9 import language_model, last_logits
from roguard.evidence import ONTOLOGY, disposition


def grouped_indices(prepared, batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError('Batch size must be between one and eight')
    groups=defaultdict(list)
    for index,(_,positions) in enumerate(prepared): groups[tuple(positions)].append(index)
    return [members[offset:offset+batch_size] for members in groups.values()
            for offset in range(0,len(members),batch_size)]


def repeated_prefix(prefix, size, repeat):
    caches=deepcopy(prefix)
    for cache in caches:
        if cache.keys is not None:
            if cache.keys.shape[0]!=1 or cache.values.shape[0]!=1:
                raise ValueError('Expected an unbatched prefix cache')
            cache.keys=repeat(cache.keys,size,axis=0)
            cache.values=repeat(cache.values,size,axis=0)
    return caches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True);parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=4);args=parser.parse_args()
    grouped_indices([],args.batch_size)
    args.output.mkdir(parents=True,exist_ok=False)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, KVCache, RotatingKVCache
    mx.set_cache_limit(512*1024*1024)
    model,tokenizer=load(args.base,adapter_path=str(args.adapter));model.set_dtype(mx.float32)
    ids=[tokenizer.encode(str(i),add_special_tokens=False) for i in range(4)]
    if any(len(v)!=1 for v in ids): raise ValueError('Class tokens must be single tokens')
    ids=[v[0] for v in ids]
    rows=[json.loads(line) for line in args.data.read_text().splitlines()]
    prepared=[aligned_tokens(tokenizer,chat(row),{field:0 for field in FIELDS}) for row in rows]
    if not rows or max(len(tokens) for tokens,_ in prepared)>1024: raise ValueError('Empty or overlength inputs')
    if len({r['case_id'] for r in rows})!=len(rows): raise ValueError('Duplicate case identities')
    common=0
    for values in zip(*(tokens[:positions[0]] for tokens,positions in prepared)):
        if len(set(values))!=1: break
        common+=1
    common=min(common,min(positions[0] for _,positions in prepared)-1)
    prefix=make_prompt_cache(model)
    if any(type(cache) not in (KVCache,RotatingKVCache) for cache in prefix):
        raise ValueError('Unsupported cache implementation for prefix repetition')
    if common: mx.eval(language_model(model).model(mx.array([prepared[0][0][:common]]),cache=prefix))
    batches=grouped_indices(prepared,args.batch_size)
    predictions=[None]*len(rows);correct={f:0 for f in FIELDS};maximum_error=0.;done=0;start=time.monotonic()
    with (args.output/'batch-progress.jsonl').open('x') as progress:
        for indices in batches:
            positions=prepared[indices[0]][1];tokens=[list(prepared[i][0]) for i in indices]
            cache=repeated_prefix(prefix,len(indices),mx.repeat);cursor=common
            interpretations=[{} for _ in indices];probabilities=[{} for _ in indices];valid=[True]*len(indices)
            for field,position in zip(FIELDS,positions):
                logits=last_logits(model,mx.array([row[cursor:position] for row in tokens]),cache=cache)[:,-1].astype(mx.float32)
                allowed=ids[:len(ONTOLOGY[field])]
                p=mx.softmax(logits[:,mx.array(allowed)],axis=-1);top=mx.argmax(logits,axis=-1);mx.eval(p,top)
                vectors=p.tolist();tops=top.tolist()
                for j,i in enumerate(indices):
                    if i<2:
                        full=model(mx.array([tokens[j][:position]]))[0,-1].astype(mx.float32)
                        check=mx.softmax(full[mx.array(allowed)]);mx.eval(check)
                        error=float(mx.max(mx.abs(check-p[j])).item());maximum_error=max(maximum_error,error)
                        if error>.001 or int(mx.argmax(full).item())!=tops[j]:
                            raise ValueError('Batched/full inference mismatch')
                    choice=max(range(len(vectors[j])),key=vectors[j].__getitem__)
                    tokens[j][position]=allowed[choice]
                    interpretations[j][field]=ONTOLOGY[field][choice];probabilities[j][field]=vectors[j]
                    valid[j] &= tops[j] in allowed
                    correct[field]+=interpretations[j][field]==rows[i]['attributes'][field]
                cursor=position
            for j,i in enumerate(indices):
                decision=int(disposition({k:{'label':v} for k,v in interpretations[j].items()})[0]!='CONTINUE') if valid[j] else None
                result=dict(case_id=rows[i]['case_id'],language=rows[i]['language'],gold=target(rows[i]),
                    predicted=decision,interpretation=interpretations[j],conditional_field_probabilities=probabilities[j],
                    valid_label_tokens=valid[j])
                predictions[i]=result;progress.write(json.dumps(result)+'\n')
            progress.flush();done+=len(indices)
            if done%25<len(indices): print(done,'of',len(rows),'decoded',flush=True)
            del cache
    if any(value is None for value in predictions): raise ValueError('Incomplete batched output')
    expected=[dict(case_id=r['case_id'],language=r['language'],gold=target(r),
        kind='conversational' if r.get('source')=='v7_training_frame_projection' else 'formal') for r in rows]
    from roguard.review_evaluation import evaluate
    record=dict(synthetic_only=True,stable_release_ready=False,base_revision=Path(args.base).name,
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),seconds=time.monotonic()-start,
        adapter_sha256=hashlib.sha256((args.adapter/'adapters.safetensors').read_bytes()).hexdigest(),
        field_accuracy={field:value/len(rows) for field,value in correct.items()},strict_evaluation=evaluate(expected,predictions),
        cached_prefix_tokens=common,cache_check_max_error=maximum_error,batch_size=args.batch_size,batches=len(batches),
        mean_batch_size=len(rows)/len(batches),conditional_probabilities_are_not_calibrated=True,reserved_probes_used=False)
    for name,values in [('expected',expected),('predictions',predictions)]:
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
    (args.output/'metrics.json').write_text(json.dumps(record,indent=2)+'\n');print('FINISHED',json.dumps(record),flush=True)


if __name__=='__main__': main()
