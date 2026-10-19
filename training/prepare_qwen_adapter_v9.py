"""Balanced, protected-input-excluded training and development for a Qwen adapter."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
from training.prepare_instruction_sft_v8 import target
from training.train_native_review_v9 import texts
from training.verify_reserved_native_v9 import PROBES


def read(p):
    raw=p.read_bytes()
    if str(p).endswith('.gz'):raw=gzip.decompress(raw)
    return [json.loads(line) for line in raw.splitlines()]


def main():
    root=Path('research/v9-stabilization');out=root/'qwen-adapter-88';out.mkdir(exist_ok=False)
    rng=random.Random(988);pool=[];dev=[];protected=set()
    for lang in ('ro','uk'):
        parent=root/('native-full-22-'+lang)
        pool.extend(read((root/'needs-repair-85/uk' if lang=='uk' else parent)/'training-history.jsonl.gz'))
        for split in ('dev','cal'):
            rows=read(parent/(split+'.jsonl.gz'));protected.update(texts(r) for r in rows)
            if split=='dev':dev.extend(rows)
    for name,digest in PROBES.values():
        p=root.parent/'v8-stabilization'/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Protected probe differs')
        protected.update(texts(dict(r,focus=r.get('focus',len(r['turns'])-1))) for r in read(p))
    # Keep the authored stress inputs for later checking, not training.
    protected.update(texts(r) for r in read(root/'needs-repair-85/uk/stress.jsonl.gz'))
    def group(r):return r['language'],r.get('source') in ('v7_training_frame_projection','distress_repair_v9','needs_repair_v9'),target(r)
    buckets=defaultdict(list);seen={}
    for row in pool:
        key=texts(row)
        if key in protected:continue
        if key in seen:
            if seen[key]!=target(row):raise ValueError('Conflicting normalized training labels')
            continue
        seen[key]=target(row);buckets[group(row)].append(row)
    train=[]
    for lang in ('ro','uk'):
        for conversational in (False,True):
            for label in (0,1):
                group_rows=buckets[(lang,conversational,label)];rng.shuffle(group_rows)
                if len(group_rows)<128:raise ValueError('Training group too small')
                train.extend(group_rows[:128])
    valid=[]
    for lang in ('ro','uk'):
        for conversational in (False,True):
            for label in (0,1):
                values=[r for r in dev if group(r)==(lang,conversational,label)];rng.shuffle(values)
                if len(values)<10:raise ValueError('Development group too small')
                valid.extend(values[:10])
    rng.shuffle(train);rng.shuffle(valid)
    if {texts(r) for r in train}&{texts(r) for r in valid}:raise ValueError('Training/development overlap')
    for name,rows in [('train',train),('valid',valid)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    config=dict(seed=988,model='mlx-community/Qwen3.5-9B-MLX-4bit',revision='938d8919941c6e7efd3c7150eff7fe9d12afa631',
        data=str(out),num_layers=1,batch_size=1,iters=1024,val_batches=80,learning_rate=1e-4,
        steps_per_report=20,steps_per_eval=256,save_every=256,max_seq_length=1024,grad_checkpoint=True,
        grad_accumulation_steps=4,lora_parameters=dict(rank=8,dropout=0.,scale=16.),fine_tune_type='lora',adapter_path=str(out/'adapters'))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    provenance=dict(training_rows=len(train),validation_rows=len(valid),synthetic_only=True,independently_reviewed=False,
        reserved_access='Input overlap exclusion only; no model inference.',calibration_used_for_selection=False,
        diagnosis_from_reused_calibration_and_screen=True,stable_release_ready=False)
    (out/'data-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance))


if __name__=='__main__':main()
