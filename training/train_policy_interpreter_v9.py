"""Continue a frozen interpreter with policy-sensitive field loss; unchanged validation."""
import argparse
import json
import hashlib
from pathlib import Path
from training.semantic_interpreter_v9 import aligned_tokens, answer, chat
from training.gemma_projection_v9 import language_model, project


class InterpretationDataset:
    def __init__(self,path,tokenizer):
        self.rows=[json.loads(l) for l in Path(path).read_text().splitlines()];self.tokenizer=tokenizer
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]
    def process(self,row):
        tokens,positions=aligned_tokens(self.tokenizer,chat(row),answer(row))
        return tokens,positions[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--base',required=True);a=p.parse_args()
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import train,evaluate,TrainingArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers,print_trainable_parameters
    config=json.loads(a.config.read_text());np.random.seed(config['seed']);mx.random.seed(config['seed'])
    mx.set_cache_limit(512*1024*1024)
    model,tok=load(a.base);model.set_dtype(mx.float32);model.freeze()
    linear_to_lora_layers(model,config['num_layers'],config['lora_parameters'])
    initial=Path(config['initial_adapter'])
    if hashlib.sha256(initial.read_bytes()).hexdigest()!=config['initial_adapter_sha256']:
        raise ValueError('Initial trained adapter differs')
    saved=mx.load(str(initial))
    expected=dict(tree_flatten(model.trainable_parameters()))
    if set(saved)!=set(expected) or any(saved[k].shape!=expected[k].shape for k in saved):
        raise ValueError('Initial adapter architecture differs')
    if any(not bool(mx.all(mx.isfinite(value)).item()) for value in saved.values()):
        raise ValueError('Initial adapter contains nonfinite values')
    model.load_weights(list(saved.items()),strict=False)
    print_trainable_parameters(model)
    datasets=[InterpretationDataset(Path(config['data'])/(s+'.jsonl'),tok) for s in ('train','valid')]
    digits=[tok.encode(str(i),add_special_tokens=False) for i in range(4)]
    if any(len(v)!=1 for v in digits):raise ValueError('Digit labels must be single tokens')
    digit_ids=[v[0] for v in digits]
    for dataset in datasets:
        for row in dataset.rows:
            tokens,start=dataset.process(row)
            if len(tokens)>config['max_seq_length'] or sum(t in digit_ids for t in tokens[start:])!=6:
                raise ValueError('Invalid aligned training example')
    from training.policy_field_weights_v9 import table
    from roguard.evidence import ONTOLOGY
    from training.semantic_interpreter_v9 import FIELDS
    policy_weights=mx.array(table(),dtype=mx.float32)
    def loss(model,batch,lengths):
        indices=mx.arange(batch.shape[0])[:,None]
        slots=mx.arange(batch.shape[1])[None,:]
        is_digit=mx.any(batch[:,:,None]==mx.array(digit_ids)[None,None,:],axis=-1)
        eligible=is_digit & (slots>=lengths[:,0,None]) & (slots<lengths[:,1,None])
        positions=mx.sort(mx.where(eligible,slots,batch.shape[1]),axis=1)[:,:6]
        hidden=language_model(model).model(batch[:,:-1])[indices,positions-1,:]
        logits=project(model,hidden)
        labels=batch[indices,positions]
        losses=nn.losses.cross_entropy(logits.astype(mx.float32),labels,reduction='none')
        if model.training:
            ordinals=mx.argmax(labels[:,:,None]==mx.array(digit_ids)[None,None,:],axis=-1)
            indices=mx.zeros((batch.shape[0],),dtype=mx.int32)
            for i,field in enumerate(FIELDS):
                indices=indices*len(ONTOLOGY[field])+ordinals[:,i]
            weights=policy_weights[indices]
            value=mx.mean(mx.sum(losses*weights,axis=-1)/mx.sum(weights,axis=-1))
        else:
            # Same unweighted full-vocabulary six-token validation as trials14/20.
            value=mx.mean(losses)
        return value,mx.array(batch.shape[0]*6)
    output=Path(config['adapter_path']);output.mkdir(exist_ok=False)
    (output/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n')
    selected=output.parent/'selected-adapter';selected.mkdir(exist_ok=False)
    architecture={k:config[k] for k in ('fine_tune_type','num_layers','lora_parameters')}
    (selected/'adapter_config.json').write_text(json.dumps(architecture,indent=2)+'\n')
    class Recorder:
        def __init__(self):self.best=float('inf');self.record={'training':[],'validation':[],'selection':'minimum full unweighted six-field development validation loss; training uses policy-sensitive field weights'}
        def save(self):
            (output.parent/'training-metrics.json').write_text(json.dumps(self.record,indent=2)+'\n')
        def on_train_loss_report(self,info):
            self.record['training'].append(info);self.save()
        def on_val_loss_report(self,info):
            if info['iteration']==0 and abs(info['val_loss']-config['initial_validation_loss'])>1e-5:
                raise ValueError('Initial unweighted validation differs from frozen parent checkpoint')
            self.record['validation'].append(info)
            if info['val_loss']<self.best:
                self.best=info['val_loss'];self.record['selected']=info
                # The callback runs at the exact evaluated parameter state,
                # before the next optimizer update, avoiding checkpoint drift.
                mx.save_safetensors(str(selected/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
            self.save()
    recorder=Recorder()
    args=TrainingArgs(batch_size=config['batch_size'],iters=config['iters'],val_batches=config['val_batches'],
        steps_per_report=config['steps_per_report'],steps_per_eval=config['steps_per_eval'],
        steps_per_save=config['save_every'],adapter_file=output/'adapters.safetensors',
        max_seq_length=config['max_seq_length'],grad_checkpoint=config['grad_checkpoint'],
        grad_accumulation_steps=config['grad_accumulation_steps'])
    train(model=model,optimizer=optim.Adam(learning_rate=config['learning_rate']),
        train_dataset=CacheDataset(datasets[0]),val_dataset=CacheDataset(datasets[1]),args=args,loss=loss,training_callback=recorder)
    mx.save_safetensors(str(output/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
    final_loss=evaluate(model,CacheDataset(datasets[1]),batch_size=config['batch_size'],
        num_batches=config['val_batches'],max_seq_length=config['max_seq_length'],loss=loss)
    recorder.on_val_loss_report({'iteration':config['iters'],'val_loss':final_loss})
    print('FINISHED policy-weighted continuation',flush=True)


if __name__=='__main__':main()
