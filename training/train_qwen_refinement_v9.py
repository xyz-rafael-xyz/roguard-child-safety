"""Adapt only the final full-attention block of a pinned Qwen model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from training.qwen_review_v9 import POLICY
from training.prepare_instruction_sft_v8 import target


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class Dataset:
    def __init__(self,path,tokenizer):
        self.rows=[json.loads(line) for line in Path(path).read_text().splitlines()];self.tokenizer=tokenizer
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]
    def process(self,row):
        focus=row.get('focus',len(row['turns'])-1)
        payload=dict(language=row['language'],prior_turns=row['turns'][:focus],focal_message=row['turns'][focus])
        text=self.tokenizer.apply_chat_template([dict(role='system',content=POLICY),dict(role='user',content=json.dumps(payload,ensure_ascii=False))],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=self.tokenizer.encode(text,add_special_tokens=False)
        label=self.tokenizer.encode(str(target(row)),add_special_tokens=False)
        if len(label)!=1:raise ValueError('Target must be one token')
        return prefix+label,len(prefix)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);a=parser.parse_args()
    study=Path('research/v9-stabilization/qwen-refinement-94')
    for path,digest in json.loads((study/'source-freeze.json').read_text()).items():
        if sha(path)!=digest:raise ValueError('Source freeze changed: '+path)
    config=json.loads((study/'config.json').read_text())
    manifest=json.loads(Path('research/v9-stabilization/qwen-control-87/model-manifest.json').read_text())
    if Path(a.base).name!=manifest['revision']:raise ValueError('Base revision differs')
    for name,value in manifest['files'].items():
        if sha(Path(a.base)/name)!=value['sha256']:raise ValueError('Base file changed')
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import train,evaluate,TrainingArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers,print_trainable_parameters
    np.random.seed(config['seed']);mx.random.seed(config['seed']);mx.set_cache_limit(512*1024*1024)
    model,tokenizer=load(a.base,tokenizer_config={'trust_remote_code':False});model.set_dtype(mx.float32);model.freeze()
    if model.layers[-1].is_linear:raise ValueError('Final trainable block must be full attention')
    linear_to_lora_layers(model,1,config['lora_parameters']);print_trainable_parameters(model)
    parameters=dict(tree_flatten(model.trainable_parameters()))
    if not parameters or any('.layers.31.' not in name for name in parameters):raise ValueError('Unexpected trainable layers')
    warm=Path('research/v9-stabilization/qwen-adapter-88/selected-adapter/adapters.safetensors')
    weights=mx.load(str(warm))
    if set(weights)!=set(parameters) or any(weights[k].shape!=v.shape or not bool(mx.all(mx.isfinite(weights[k])).item()) for k,v in parameters.items()):
        raise ValueError('Warm adapter schema differs')
    model.load_weights(list(weights.items()),strict=False)
    (study/'warm-start.json').write_text(json.dumps(dict(parent='qwen-adapter-88',adapter_sha256=sha(warm)),indent=2)+'\n')
    datasets=[Dataset(study/(s+'.jsonl'),tokenizer) for s in ['train','valid']]
    for data in datasets:
        for row in data.rows:
            tokens,start=data.process(row)
            if len(tokens)>config['max_seq_length'] or start!=len(tokens)-1:raise ValueError('Invalid target alignment or excess length')
    def loss(model,batch,lengths):
        # Frozen recurrent blocks use their inference kernel. Only the final
        # full-attention block participates in parameter differentiation.
        if model.training:
            for layer in model.layers[:-1]:layer.eval()
        positions=lengths[:,0]-1;indices=mx.arange(batch.shape[0])
        hidden=model.language_model.model(batch[:,:-1])[indices,positions,:][:,None,:]
        logits=model.language_model.lm_head(hidden)[:,0,:].astype(mx.float32)
        labels=batch[indices,positions+1]
        return nn.losses.cross_entropy(logits,labels,reduction='mean'),mx.array(batch.shape[0])
    reference_loss=loss
    from training.qwen_cached_loss_v9 import make_loss
    sequences=[data.process(row)[0] for data in datasets for row in data.rows]
    loss,common=make_loss(model,sequences)
    import time
    checks=[]
    for language in ('ro','uk'):
        row=next(r for r in datasets[1].rows if r['language']==language)
        tokens,start=datasets[1].process(row);batch=mx.array([tokens]);lengths=mx.array([[start,len(tokens)]])
        results=[]
        for method in (reference_loss,loss):
            began=time.monotonic();value,grad=nn.value_and_grad(model,method)(model,batch,lengths);mx.eval(value,grad)
            results.append((float(value[0].item()),dict(tree_flatten(grad)),time.monotonic()-began))
        absolute=abs(results[0][0]-results[1][0])
        gradient=max(float(mx.max(mx.abs(results[0][1][k]-results[1][1][k])).item())/max(1.,float(mx.max(mx.abs(results[0][1][k])).item())) for k in results[0][1])
        checks.append(dict(language=language,loss_error=absolute,scaled_gradient_error=gradient,
            reference_seconds=results[0][2],cached_seconds=results[1][2]))
    report=dict(common_prefix_tokens=common,checks=checks,passed=all(c['loss_error']<=.001 and c['scaled_gradient_error']<=.001 for c in checks))
    (study/'cache-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise ValueError('Frozen-prefix loss/gradient parity failed before optimization')
    output=study/'adapters';output.mkdir(exist_ok=False)
    architecture={k:config[k] for k in ['fine_tune_type','num_layers','lora_parameters']}
    (output/'adapter_config.json').write_text(json.dumps(architecture,indent=2)+'\n')
    selected=study/'selected-adapter';selected.mkdir(exist_ok=False)
    (selected/'adapter_config.json').write_text(json.dumps(architecture,indent=2)+'\n')
    class Recorder:
        def __init__(self):self.best=float('inf');self.record=dict(training=[],validation=[],selection='Minimum fixed80-development full-vocabulary one-token loss',reserved_probes_used=False,stable_release_ready=False)
        def save(self):(study/'training-metrics.json').write_text(json.dumps(self.record,indent=2)+'\n')
        def on_train_loss_report(self,info):
            if not math.isfinite(info['train_loss']):raise ValueError('Nonfinite training loss')
            self.record['training'].append(info);self.save()
        def on_val_loss_report(self,info):
            if not math.isfinite(info['val_loss']):raise ValueError('Nonfinite validation loss')
            self.record['validation'].append(info)
            snapshot=study/'validation-checkpoints'/str(info['iteration']);snapshot.mkdir(parents=True,exist_ok=False)
            (snapshot/'adapter_config.json').write_text(json.dumps(architecture,indent=2)+'\n')
            params=dict(tree_flatten(model.trainable_parameters()))
            if any(not bool(mx.all(mx.isfinite(v)).item()) for v in params.values()):raise ValueError('Nonfinite adapter weights')
            mx.save_safetensors(str(snapshot/'adapters.safetensors'),params)
            self.record.setdefault('checkpoints',[]).append(dict(iteration=info['iteration'],val_loss=info['val_loss'],sha256=sha(snapshot/'adapters.safetensors')))
            if info['val_loss']<self.best:
                self.best=info['val_loss'];self.record['selected']=info
                mx.save_safetensors(str(selected/'adapters.safetensors'),params)
            self.save()
    recorder=Recorder()
    args=TrainingArgs(batch_size=1,iters=config['iters'],val_batches=80,steps_per_report=20,steps_per_eval=256,
        steps_per_save=256,adapter_file=output/'adapters.safetensors',max_seq_length=1024,grad_checkpoint=True,grad_accumulation_steps=4)
    train(model=model,optimizer=optim.Adam(learning_rate=optim.cosine_decay(config['learning_rate'],config['iters']//4,end=config['learning_rate']/10)),train_dataset=CacheDataset(datasets[0]),
        val_dataset=CacheDataset(datasets[1]),args=args,loss=loss,training_callback=recorder)
    final_loss=evaluate(model,CacheDataset(datasets[1]),batch_size=1,num_batches=80,max_seq_length=1024,loss=loss)
    recorder.on_val_loss_report(dict(iteration=config['iters'],val_loss=final_loss))
    print('FINISHED Qwen refinement training',flush=True)


if __name__=='__main__':main()
