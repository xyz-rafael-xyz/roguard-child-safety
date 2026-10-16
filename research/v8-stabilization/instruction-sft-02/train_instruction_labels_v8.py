"""Train only the supervised class token, not template scaffolding or padding."""
import argparse
import json
from pathlib import Path
from training.evaluate_instruction_sft_v8 import completion_prefix


class LabelDataset:
    def __init__(self, path, tokenizer):
        self.rows=[json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines()]
        self.tokenizer=tokenizer
        self.ids=[tokenizer.encode(str(i),add_special_tokens=False) for i in (0,1)]
        if any(len(v)!=1 for v in self.ids):raise ValueError('Class labels must be single tokens')
        self.ids=[v[0] for v in self.ids]

    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]
    def process(self,row):
        chat=row['messages']
        if chat[-1]['role']!='assistant' or chat[-1]['content'] not in ('0','1'):
            raise ValueError('Expected a binary supervised answer')
        prefix=completion_prefix(self.tokenizer,chat[:-1],self.ids)
        return prefix+[self.ids[int(chat[-1]['content'])]],len(prefix)


def classification_loss(model,batch,lengths):
    import mlx.core as mx
    import mlx.nn as nn
    positions=lengths[:,0]
    indices=mx.arange(batch.shape[0])
    hidden=model.model(batch[:,:-1])[indices,positions-1,:][:,None,:]
    logits=(model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings else model.lm_head(hidden))[:,0,:]
    labels=batch[indices,positions]
    loss=nn.losses.cross_entropy(logits.astype(mx.float32),labels,reduction='mean')
    return loss,mx.array(batch.shape[0])


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--base',required=True);a=p.parse_args()
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import train,TrainingArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers,print_trainable_parameters
    config=json.loads(a.config.read_text(encoding='utf-8'))
    np.random.seed(config['seed']);mx.random.seed(config['seed'])
    model,tok=load(a.base);model.freeze()
    linear_to_lora_layers(model,config['num_layers'],config['lora_parameters'])
    model.load_weights(config['resume_adapter_file'],strict=False)
    print_trainable_parameters(model)
    trainset=LabelDataset(Path(config['data'])/'train.jsonl',tok)
    validset=LabelDataset(Path(config['data'])/'valid.jsonl',tok)
    # Eager validation before the first update: no truncation and exactly one
    # supervised label position. End markers and padding have no loss.
    for dataset in (trainset,validset):
        for row in dataset.rows:
            tokens,position=dataset.process(row)
            if position<1 or position!=len(tokens)-1 or len(tokens)>config['max_seq_length']:
                raise ValueError('Invalid supervised label position or overlength input')
    output=Path(config['adapter_path']);output.mkdir(parents=True,exist_ok=False)
    (output/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    args=TrainingArgs(batch_size=config['batch_size'],iters=config['iters'],val_batches=config['val_batches'],
        steps_per_report=config['steps_per_report'],steps_per_eval=config['steps_per_eval'],steps_per_save=config['save_every'],
        adapter_file=output/'adapters.safetensors',max_seq_length=config['max_seq_length'],
        grad_checkpoint=config['grad_checkpoint'],grad_accumulation_steps=config['grad_accumulation_steps'])
    train(model=model,optimizer=optim.Adam(learning_rate=config['learning_rate']),
        train_dataset=CacheDataset(trainset),val_dataset=CacheDataset(validset),args=args,loss=classification_loss)
    mx.save_safetensors(str(output/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
    print('FINISHED label-only training',flush=True)

if __name__=='__main__':main()
