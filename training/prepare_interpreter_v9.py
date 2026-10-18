"""Prepare balanced training-only interpretations for the Gemma 12B pilot."""
import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path
from training.prepare_instruction_sft_v8 import target
from training.review_data_v8 import input_key
from training.natural_contrasts_v9 import generate as natural
from training.temporal_contrasts_v9 import generate as temporal
from training.semantic_interpreter_v9 import chat, answer, aligned_tokens, POLICY


def read(path):
    return [json.loads(l) for l in gzip.decompress(path.read_bytes()).decode().splitlines()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rng=random.Random(914);pool=[];root=Path('research/v8-stabilization')
    for name,lang in (('review-09-ro','ro'),('review-08-uk','uk')):
        pool.extend(r for r in read(root/name/'training-history.jsonl.gz') if r['language']==lang)
    chosen=[];seen=set()
    for lang in ('ro','uk'):
        for native in (False,True):
            for label in (0,1):
                group=[r for r in pool if r['language']==lang and (r.get('source')=='v7_training_frame_projection')==native and target(r)==label]
                rng.shuffle(group);count=0
                for row in group:
                    key=input_key(row)
                    if key in seen:continue
                    chosen.append(row);seen.add(key);count+=1
                    if count==200:break
                if count!=200:raise ValueError('Missing historical training group')
    new=natural()+temporal()
    for lang in ('ro','uk'):
        for label in (0,1):
            group=[r for r in new if r['language']==lang and target(r)==label];rng.shuffle(group);count=0
            for row in group:
                key=input_key(row)
                if key in seen:continue
                chosen.append(row);seen.add(key);count+=1
                if count==100:break
            if count!=100:raise ValueError('Missing new-content training group')
    for split in ('dev','cal'):
        if seen & {input_key(r) for r in read(root/'review-09-ro'/(split+'.jsonl.gz'))}:raise ValueError('Evaluation overlap')
    # Development screen is reused for model selection; no fresh probe is read.
    dev=[json.loads(l) for l in (root/'instruction-sft-01/screen.jsonl').read_text().splitlines()]
    if seen & set(map(input_key,dev)):raise ValueError('Screen overlap')
    rng.shuffle(chosen)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    lengths=[]
    for row in chosen+dev[:80]:
        tokens,positions=aligned_tokens(tok,chat(row),answer(row));lengths.append(len(tokens))
        digit_ids={tok.encode(str(i),add_special_tokens=False)[0] for i in range(4)}
        if sum(t in digit_ids for t in tokens[positions[0]:])!=6:raise ValueError('Unexpected digit in output scaffold')
    if max(lengths)>1024:raise ValueError('Input exceeds pilot token budget')
    a.output.mkdir(exist_ok=False)
    for name,rows in (('train',chosen),('valid',dev[:80])):
        (a.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (a.output/'policy.txt').write_text(POLICY+'\n')
    config=dict(seed=914,model='mlx-community/gemma-3-text-12b-it-4bit',revision=Path(a.base).name,
        data=str(a.output),num_layers=8,batch_size=1,iters=len(chosen),val_batches=4,learning_rate=0.00002,
        steps_per_report=20,steps_per_eval=500,save_every=500,max_seq_length=1024,grad_checkpoint=True,
        grad_accumulation_steps=4,lora_parameters=dict(rank=8,dropout=0.0,scale=16.0),adapter_path=str(a.output/'adapters'))
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    protocol=dict(synthetic_only=True,train_rows=len(chosen),valid_rows=80,max_tokens=max(lengths),
        purpose='joint semantic interpretation; classify all six fields with class-token-only loss',
        comparison='unadapted and adapted model under identical structured prompt on the fixed 400-case development screen',
        reserved_probes_used=False,stable_release_ready=False,selection='development only, followed by full regression and fresh gates',
        file_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file()})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps(protocol),flush=True)


if __name__=='__main__':main()
