"""Adapt shared-KV attention on TRAIN; retain every development checkpoint."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.gemma4_attention import replay
from roguard.gemma4_review import _Engine, _score
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.cache_semantic_tail_v10 import AddedReader
from training.composite_tail_cache_v9 import CompositeReader
from training.native_fields_v9 import SIZES, policy_mass, targets
from training.prepare_instruction_sft_v8 import target
from training.separation_frontier_v9 import frontier
from training.shared_tail_cache_v9 import batch_states, to_mlx


class MixedReader:
    def __init__(self, study, private, development_study, development_private, cache_root):
        roots = [cache_root/name for name in
            ('shared-tail-corpus-164', 'grounding-tail-corpus-174',
             'referent-tail-corpus-176', 'reported-tail-corpus-195')]
        self.old = CompositeReader(roots); self.added = AddedReader(study, private)
        self.boundary = len(self.old.entries)
        self.development = AddedReader(development_study, development_private)
        if any(r['partition']!='train' for r in self.added.entries) or any(r['partition']!='dev' for r in self.development.entries):
            raise ValueError('Additional cache partitions differ from the registered plan')
        self.development_boundary = self.boundary+len(self.added.entries)
        self.entries = self.old.entries+self.added.entries+self.development.entries
        hashes = [r['token_sha256'] for r in self.entries]
        if len(hashes) != len(set(hashes)):
            raise ValueError('Repeated token input across cache corpora')

    def read_many(self, indices):
        groups = defaultdict(list); output = [None]*len(indices)
        for position, index in enumerate(indices):
            if type(index) is not int or not 0 <= index < len(self.entries):
                raise ValueError('Invalid cache index')
            group = 2 if index>=self.development_boundary else 1 if index>=self.boundary else 0
            groups[group].append((position,index))
        for group, positions in groups.items():
            reader = (self.old,self.added,self.development)[group]
            offset = (0,self.boundary,self.development_boundary)[group]
            values = reader.read_many([i-offset for _,i in positions])
            for (position, _), state in zip(positions, values):
                output[position] = state
        return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'candidate', 'added_study', 'added_private', 'development_study',
                'development_private', 'features', 'feature_study', 'study', 'private'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    p.add_argument('--cache-root',type=Path,default=Path('research/v9-stabilization'))
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite attention training')
    amendment = Path('research/v10-evidence/protocol-amendment-11.json')
    configuration = json.loads(amendment.read_text())['registered_configuration']
    if configuration['epochs'] != 1 or configuration['batch_size'] != 8:
        raise ValueError('This registered implementation requires one epoch and batch size eight')
    extraction = json.loads((a.feature_study/'report.json').read_text())
    if not extraction['complete']:
        raise ValueError('Teacher representations are incomplete')
    for lang, record in extraction['languages'].items():
        for suffix, expected in record['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix)) != expected:
                raise ValueError('Teacher binding differs')
    reader = MixedReader(a.added_study, a.added_private, a.development_study, a.development_private,a.cache_root); entries = reader.entries
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    protocol = {'format': 'roguard-semantic-tail-fit-1', 'configuration': configuration,
        'source_sha256': file_sha(Path(__file__)), 'amendment_sha256': file_sha(amendment),
        'additional_cache_report_sha256': file_sha(a.added_study/'report.json'),
        'development_cache_report_sha256': file_sha(a.development_study/'report.json'),
        'development_amendment_sha256': file_sha(Path('research/v10-evidence/protocol-amendment-12.json')),
        'teacher_report_sha256': file_sha(a.feature_study/'report.json'),
        'old_cache_parents': reader.old.parents, 'calibration_or_test_loaded': False,
        'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file, save_file
    report = {'complete': False, 'languages': {}, 'stable_release_ready': False,
              'calibration_or_test_loaded': False}; started = time.monotonic()
    def persist():
        report['seconds'] = time.monotonic()-started
        (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for lang in ('ro', 'uk'):
        component = frozen.models[lang]; engine = _Engine(*component._args); model = engine.model
        model.heads['uk' if lang == 'ro' else 'ro'].freeze()
        parameters = dict(tree_flatten(model.trainable_parameters()))
        for name in parameters:
            if name.startswith('heads.'+lang+'.'):
                continue
            match = re.fullmatch(r'base\.model\.layers\.(\d+)\.self_attn\.(q_proj|o_proj)\.lora_[ab]', name)
            if not match or int(match[1]) < 24:
                raise ValueError('Unregistered trainable parameter: '+name)
        original = load_file(a.candidate/'actual'/(lang+'-selected.safetensors'))
        metadata = read_jsonl(a.features/(lang+'-rows.jsonl'))
        teacher_logits = np.load(a.features/(lang+'-logits.npy'), mmap_mode='r')
        teacher_lookup = {r['token_sha256']: i for i,r in enumerate(metadata)}
        train = [i for i,r in enumerate(entries) if r['partition'] == 'train' and r['row']['language'] == lang]
        dev = [i for i,r in enumerate(entries) if r['partition'] == 'dev' and r['row']['language'] == lang]
        local = [entries[i] for i in train]
        labels = np.array(targets([r['row'] for r in local]), dtype=np.int32)
        y = np.array([target(r['row']) for r in local], dtype=np.int32)
        teacher = np.stack([teacher_logits[teacher_lookup[r['token_sha256']]] for r in local])
        for r, fields, review in zip(local, labels, y):
            record = metadata[teacher_lookup[r['token_sha256']]]
            if record['partition'] != 'train' or record['fields'] != fields.tolist() or record['gold'] != int(review):
                raise ValueError('Training teacher and supervision differ')
        strata = [(i >= reader.boundary, int(gold)) for i,gold in zip(train,y)]
        counts = Counter(strata)
        if len(counts) != 4:
            raise ValueError('Expected old/new by binary-label strata')
        row_weights = np.array([len(train)/(4*counts[s]) for s in strata], dtype=np.float32)
        field_weights = []
        for j, size in enumerate(SIZES):
            frequencies = np.bincount(labels[:,j], minlength=size)
            values = np.minimum(4., np.sqrt(len(train)/np.maximum(1, frequencies)/size))
            values /= np.mean(values[labels[:,j]])
            field_weights.append(mx.array(values.astype(np.float32)))
        record = {'training_rows': len(train), 'development_rows': len(dev),
            'trainable_parameters': sum(v.size for v in parameters.values()), 'checkpoints': [], 'selected': None}
        report['languages'][lang] = record
        def evaluate(epoch):
            model.eval(); values = []
            with ThreadPoolExecutor(max_workers=1) as pool:
                chunks = [dev[start:start+32] for start in range(0,len(dev),32)]
                future = pool.submit(reader.read_many, chunks[0])
                for block, chosen in enumerate(chunks):
                    states = future.result()
                    if block+1 < len(chunks):
                        future = pool.submit(reader.read_many, chunks[block+1])
                    groups = defaultdict(list); output = [None]*len(states)
                    for i,state in enumerate(states): groups[state['length']].append(i)
                    for indices in groups.values():
                        for start in range(0,len(indices),8):
                            which = indices[start:start+8]
                            hidden = replay(model.base, batch_states([to_mlx(states[i]) for i in which]))[:,0,:]
                            z = model.heads[lang](hidden); mx.eval(z)
                            for position,i in enumerate(which): output[i] = np.asarray(z)[position].copy()
                    values.extend(output); del states; mx.clear_cache()
            z = np.stack(values).astype(np.float64)
            if not np.isfinite(z).all(): raise ValueError('Nonfinite development logits')
            def softmax(v):
                exp = np.exp(v-v.max(axis=1,keepdims=True)); return exp/exp.sum(axis=1,keepdims=True)
            factors = [softmax(v) for v in np.split(z[:,:sum(SIZES)],np.cumsum(SIZES)[:-1],axis=1)]
            gold_fields = np.array(targets([entries[i]['row'] for i in dev]))
            predictions = np.column_stack([v.argmax(axis=1) for v in factors])
            joint_errors = int(np.any(predictions != gold_fields,axis=1).sum())
            ce = float(np.mean([-np.log(np.maximum(v[np.arange(len(dev)),gold_fields[:,j]],1e-30)).mean()
                               for j,v in enumerate(factors)]))
            modes = {'binary':softmax(z[:,-2:])[:,1], 'policy':policy_mass(factors),
                     'coupled_policy':policy_mass(factors,component._spec['couplings'])}
            choices = []
            for mode,scores in modes.items():
                if epoch == 0 and mode != component._spec['mode']: continue
                population = [{'case_id':entries[i]['row']['case_id'], 'language':lang,
                    'group':entries[i]['group'], 'gold':target(entries[i]['row']), 'score':float(score)}
                    for i,score in zip(dev,scores)]
                boundary = frontier(population)
                threshold = component._spec['threshold'] if epoch == 0 else boundary['threshold']
                for row in population: row['predicted'] = int(row['score'] >= threshold)
                errors = sum(r['predicted'] != r['gold'] for r in population)
                choices.append((errors,-boundary['score_margin'],mode,threshold,population))
            errors,negative_margin,mode,threshold,population = min(choices,key=lambda v:v[:2])
            if epoch == 0:
                reference_scores = [_score(entries[i]['reference_logits'] if 'reference_logits' in entries[i]
                    else teacher_logits[teacher_lookup[entries[i]['token_sha256']]],component._spec) for i in dev]
                error = max(abs(r['score']-v) for r,v in zip(population,reference_scores))
                record['initial_maximum_score_error'] = error
                old_errors = sum(r['predicted']!=r['gold'] for i,r in zip(dev,population) if i<reader.boundary)
                record['initial_original_development_errors'] = old_errors
                if error > 1e-5 or old_errors != 0: raise ValueError('Released development baseline parity failed')
            tensors = dict(original)
            tensors.update({k:np.asarray(v).copy() for k,v in tree_flatten(model.trainable_parameters())})
            destination = a.private/lang; destination.mkdir(exist_ok=True)
            checkpoint = destination/('epoch-'+str(epoch)+'.safetensors'); save_file(tensors,str(checkpoint))
            rank = [errors,joint_errors,ce,negative_margin]
            summary = {'epoch':epoch, 'binary_errors':errors, 'joint_argmax_errors':joint_errors,
                'field_errors':(predictions!=gold_fields).sum(axis=0).tolist(),
                'mean_field_cross_entropy':ce, 'mode':mode, 'threshold':threshold,
                'score_margin':-negative_margin, 'selection_rank':rank,
                'adapter_sha256':file_sha(checkpoint)}
            record['checkpoints'].append(summary)
            if record['selected'] is None or rank < record['selected']['selection_rank']:
                record['selected'] = summary
                selected_path = a.private/(lang+'-selected.safetensors')
                selected_path.write_bytes(checkpoint.read_bytes())
                spec = dict(component._spec,adapter_sha256=file_sha(selected_path),mode=mode,threshold=threshold)
                (a.study/(lang+'-review.json')).write_text(json.dumps(spec,indent=2)+'\n')
                write_jsonl(a.study/(lang+'-development-predictions.jsonl'),population)
            persist(); print(json.dumps({'language':lang,**summary}),flush=True); model.train()
        evaluate(0)
        mx.random.seed(configuration['seed']); rng = np.random.default_rng(configuration['seed'])
        order = rng.permutation(len(train)); batches = [order[start:start+8].tolist() for start in range(0,len(train),8)]
        optimizer = optim.AdamW(learning_rate=configuration['learning_rate'],weight_decay=configuration['weight_decay'])
        def loss_fn(current,states,gold,fields,weights,teacher_values):
            # Serial attention calls avoid the documented Metal batched-backward issue.
            logits = mx.concatenate([current.heads[lang](replay(current.base,state)[:,0,:]) for state in states],axis=0)
            offset = 0; terms = []; divergences = []
            for j,size in enumerate(SIZES):
                values = logits[:,offset:offset+size]; old = teacher_values[:,offset:offset+size]
                terms.append(nn.losses.cross_entropy(values,fields[:,j])*field_weights[j][fields[:,j]])
                old_logp = old-mx.logsumexp(old,axis=-1,keepdims=True)
                logp = values-mx.logsumexp(values,axis=-1,keepdims=True)
                divergences.append((mx.exp(old_logp)*(old_logp-logp)).sum(axis=-1)); offset += size
            old = teacher_values[:,-2:]; values = logits[:,-2:]
            old_logp = old-mx.logsumexp(old,axis=-1,keepdims=True)
            logp = values-mx.logsumexp(values,axis=-1,keepdims=True)
            divergences.append((mx.exp(old_logp)*(old_logp-logp)).sum(axis=-1))
            return (weights*(configuration['binary_ce_weight']*nn.losses.cross_entropy(values,gold)
                +configuration['semantic_ce_weight']*mx.stack(terms).mean(axis=0)
                +configuration['teacher_kl_weight']*mx.stack(divergences).mean(axis=0))).mean()
        value_grad = nn.value_and_grad(model,loss_fn); losses = []; training_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(reader.read_many,[train[i] for i in batches[0]])
            for step,chosen in enumerate(batches):
                states = [to_mlx(s) for s in future.result()]
                if step+1 < len(batches): future = pool.submit(reader.read_many,[train[i] for i in batches[step+1]])
                warm = max(1,int(.05*len(batches)))
                scale = min((step+1)/warm,max(.1,(len(batches)-step)/(len(batches)-warm)))
                optimizer.learning_rate = configuration['learning_rate']*scale
                loss,gradient = value_grad(model,states,mx.array(y[chosen]),mx.array(labels[chosen]),
                    mx.array(row_weights[chosen]),mx.array(teacher[chosen]))
                gradient,norm = optim.clip_grad_norm(gradient,configuration['gradient_norm_clip'])
                mx.eval(loss,norm,gradient)
                if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):
                    raise ValueError('Nonfinite training loss or gradient')
                optimizer.update(model,gradient); mx.eval(model.trainable_parameters(),optimizer.state)
                losses.append(float(loss.item())); del states,gradient; mx.clear_cache()
                if step%100 == 0:
                    progress = {'language':lang,'step':step+1,'total_steps':len(batches),
                                'loss':losses[-1],'seconds':time.monotonic()-training_started}
                    (a.study/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                    print(json.dumps(progress),flush=True)
                if step+1 in (len(batches)//2,len(batches)):
                    evaluate(.5 if step+1 < len(batches) else 1)
        record.update(complete=True,mean_training_loss=float(np.mean(losses)),training_seconds=time.monotonic()-training_started)
        persist(); del engine,model,optimizer,value_grad,parameters
        import gc
        gc.collect(); mx.clear_cache()
    report['complete'] = True; persist()


if __name__ == '__main__':
    main()
