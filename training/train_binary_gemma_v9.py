"""Matched Gemma binary-review training with one supervised class token."""
import argparse
import json
from pathlib import Path
from training.evaluate_gemma_review_v9 import classifier_tokens
from training.prepare_instruction_sft_v8 import target
from training.gemma_projection_v9 import language_model, project


class Dataset:
    def __init__(self,path,tokenizer):
        self.rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
        self.tokenizer=tokenizer
        ids=[tokenizer.encode(str(i),add_special_tokens=False) for i in (0,1)]
        if any(len(v)!=1 for v in ids):raise ValueError('Class labels must be single tokens')
        self.ids=[v[0] for v in ids]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]
    def process(self,row):
        prefix=classifier_tokens(self.tokenizer,row,self.ids)
        return prefix+[self.ids[target(row)]],len(prefix)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import train,evaluate,TrainingArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers,print_trainable_parameters
    config=json.loads(args.config.read_text());np.random.seed(config['seed']);mx.random.seed(config['seed'])
    mx.set_cache_limit(512*1024*1024)
    model,tokenizer=load(args.base);model.set_dtype(mx.float32);model.freeze()
    linear_to_lora_layers(model,config['num_layers'],config['lora_parameters']);print_trainable_parameters(model)
    datasets=[Dataset(Path(config['data'])/(split+'.jsonl'),tokenizer) for split in ('train','valid')]
    for dataset in datasets:
        for row in dataset.rows:
            tokens,start=dataset.process(row)
            if start<1 or start!=len(tokens)-1 or len(tokens)>config['max_seq_length']:
                raise ValueError('Invalid supervised class position or overlength input')
    def loss(model,batch,lengths):
        indices=mx.arange(batch.shape[0]);positions=lengths[:,0]
        hidden=language_model(model).model(batch[:,:-1])[indices,positions-1,:][:,None,:]
        logits=project(model,hidden)[:,0,:]
        value=nn.losses.cross_entropy(logits.astype(mx.float32),batch[indices,positions],reduction='mean')
        return value,mx.array(batch.shape[0])
    output=Path(config['adapter_path']);output.mkdir(exist_ok=False)
    (output/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n')
    selected=output.parent/'selected-adapter';selected.mkdir(exist_ok=False)
    architecture={k:config[k] for k in ('fine_tune_type','num_layers','lora_parameters')}
    (selected/'adapter_config.json').write_text(json.dumps(architecture,indent=2)+'\n')
    class Recorder:
        def __init__(self):self.best=float('inf');self.record=dict(training=[],validation=[],selection='Minimum full 80-row binary class-token validation loss; earlier tie.')
        def save(self):(output.parent/'training-metrics.json').write_text(json.dumps(self.record,indent=2)+'\n')
        def on_train_loss_report(self,info):self.record['training'].append(info);self.save()
        def on_val_loss_report(self,info):
            self.record['validation'].append(info)
            if info['val_loss']<self.best:
                self.best=info['val_loss'];self.record['selected']=info
                mx.save_safetensors(str(selected/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
            self.save()
    recorder=Recorder()
    options=TrainingArgs(batch_size=config['batch_size'],iters=config['iters'],val_batches=config['val_batches'],
                        steps_per_report=config['steps_per_report'],steps_per_eval=config['steps_per_eval'],
                        steps_per_save=config['save_every'],adapter_file=output/'adapters.safetensors',
                        max_seq_length=config['max_seq_length'],grad_checkpoint=config['grad_checkpoint'],
                        grad_accumulation_steps=config['grad_accumulation_steps'])
    train(model=model,optimizer=optim.Adam(learning_rate=config['learning_rate']),train_dataset=CacheDataset(datasets[0]),
          val_dataset=CacheDataset(datasets[1]),args=options,loss=loss,training_callback=recorder)
    mx.save_safetensors(str(output/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
    final=evaluate(model,CacheDataset(datasets[1]),batch_size=config['batch_size'],num_batches=config['val_batches'],
                   max_seq_length=config['max_seq_length'],loss=loss)
    recorder.on_val_loss_report(dict(iteration=config['iters'],val_loss=final))
    print('FINISHED binary Gemma training',flush=True)


if __name__=='__main__':main()
