"""CPU-only fixed-feature/head gradient diagnostic; no encoder, fitting or DEV gradients."""
import argparse
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from roguard.trajectory_data import file_sha
from roguard.trajectory_model import RESPONSE_LABELS
from training.train_large_response_attention_v10 import load_inputs
from training.train_large_response_v10 import training_weights
from training.train_mixed_response_attention_v10 import prepare_sampler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite gradient diagnostic')
    evidence=a.root/'research/v10-evidence'
    command=json.loads((a.private/'mixed-large-response-attention-65-command.json').read_text())
    args=Namespace(**{command[i][2:].replace('-','_'):Path(command[i+1]) for i in range(2,len(command),2)})
    rows,x,train,dev,identities,parent,data,bindings=load_inputs(args)
    preflight=json.loads((evidence/'mixed-large-response-attention-preflight-65/report.json').read_text())
    if bindings!=preflight['bindings']:raise ValueError('Bound TRAIN/features changed')
    plan=json.loads(args.amendment.read_text())
    if not plan['fresh_response_head'] or plan['optimization']['teacher_kl_weight']!=0:raise ValueError('Unsupported diagnostic regime')
    mixed=prepare_sampler(rows,train,plan['optimization'],plan['sampling'])
    uniform_weights=training_weights(rows,train)
    checkpoint=a.private/'large-response-attention-59/epoch-0.safetensors'
    reference=json.loads((evidence/'large-response-attention-59/progress.json').read_text())['epochs'][0]
    if file_sha(checkpoint)!=reference['adapter_sha256']:raise ValueError('Initialization checkpoint changed')
    arrays=load_file(checkpoint);w=arrays['head.last.weight'].astype(np.float64);b=arrays['head.last.bias'].astype(np.float64)
    labels=np.array([RESPONSE_LABELS.index(r['gold']) for r in rows])
    if any(rows[i]['partition']!='train' for i in train) or set(train)&set(dev):raise ValueError('Invalid gradient population')
    def gradient(indices,weights):
        features=x[indices].astype(np.float64);z=features@w.T+b;z-=z.max(axis=1,keepdims=True)
        prob=np.exp(z);prob/=prob.sum(axis=1,keepdims=True)
        prob[np.arange(len(indices)),labels[indices]]-=1
        residual=prob*(weights[indices,None]/len(indices))
        return np.concatenate(((residual.T@features).ravel(),residual.sum(axis=0)))
    # Use a separate RNG for a directional finite-difference check; sampling RNGs stay exact.
    direction=np.random.default_rng(2065).normal(size=w.size+b.size);direction/=np.linalg.norm(direction)
    dw=direction[:w.size].reshape(w.shape);db=direction[w.size:]
    def loss(indices,weights,scale):
        z=x[indices].astype(np.float64)@(w+scale*dw).T+(b+scale*db)
        maxima=z.max(axis=1);lse=maxima+np.log(np.exp(z-maxima[:,None]).sum(axis=1))
        return float(np.mean(weights[indices]*(lse-z[np.arange(len(indices)),labels[indices]])))
    uniform_order=np.random.default_rng(1059).permutation(train)
    uniform_batches=[uniform_order[i:i+8] for i in range(0,len(train),8)]
    mixed_batches=mixed.batches(np.random.default_rng(1059))
    finite_checks=[]
    for batch,weights in [(uniform_batches[0],uniform_weights),(mixed_batches[0],mixed.weights)]:
        analytic=float(gradient(batch,weights)@direction);epsilon=1e-5
        numeric=(loss(batch,weights,epsilon)-loss(batch,weights,-epsilon))/(2*epsilon)
        if not np.isclose(analytic,numeric,rtol=1e-6,atol=1e-8):raise ValueError('Head gradient finite-difference check failed')
        finite_checks.append({'analytic':analytic,'finite_difference':numeric,'absolute_error':abs(analytic-numeric)})
    # The exact finite-TRAIN equal-group objective provides a common fixed reference.
    full=np.zeros(w.size+b.size)
    for offset in range(0,len(train),256):
        subset=np.array(train[offset:offset+256]);full+=gradient(subset,uniform_weights)*len(subset)/len(train)
    summaries={}
    for name,batches,weights in [('uniform',uniform_batches,uniform_weights),('mixed',mixed_batches,mixed.weights)]:
        summed=np.zeros_like(full);second=0.;norms=[];weighted=np.zeros_like(full);presentations=0
        for batch in batches:
            g=gradient(batch,weights)
            if not np.isfinite(g).all():raise ValueError('Nonfinite head gradient')
            norm=float(np.linalg.norm(g));norms.append(norm);summed+=g;second+=norm*norm
            weighted+=g*len(batch);presentations+=len(batch)
        mean=summed/len(batches);variance=second/len(batches)-float(mean@mean)
        if name=='uniform' and not np.allclose(weighted/presentations,full,rtol=1e-10,atol=1e-10):raise ValueError('Partitioned uniform objective differs')
        summaries[name]={'batches':len(batches),'presentations':presentations,
                         'head_gradient_mean_norm':float(np.linalg.norm(mean)),
                         'head_gradient_variance_trace':variance,'head_gradient_second_moment':second/len(batches),
                         'head_gradient_norm_quantiles':dict(zip(('min','p05','median','p95','max'),np.quantile(norms,[0,.05,.5,.95,1]).tolist())),
                         'head_gradient_norm_above_one_fraction':sum(n>1 for n in norms)/len(norms),
                         'unweighted_batch_mean_difference_from_full_objective_norm':float(np.linalg.norm(mean-full))}
    report={'complete':True,'source_sha256':file_sha(Path(__file__)),'scope':'One unupdated first-pass sampler comparison using the completed59 fresh epoch0 linear head and cached TRAIN features, NumPy float64 CE gradients only. No optimizer updates or attention gradients.',
            'checkpoint_sha256':file_sha(checkpoint),'feature_sha256':bindings['large_features_sha256'],
            'bindings':bindings,'sampler_source_sha256':file_sha(a.root/'training/response_sampling_v10.py'),
            'head_shape':list(w.shape),'train_rows':len(train),'gradient_parameters':int(w.size+b.size),
            'finite_difference_checks':finite_checks,'full_train_objective_head_gradient_norm':float(np.linalg.norm(full)),
            'summaries':summaries,'mixed_to_uniform_head_gradient_variance_ratio':summaries['mixed']['head_gradient_variance_trace']/summaries['uniform']['head_gradient_variance_trace'],
            'attention_gradients_measured':False,'training_trajectory_or_accuracy_measured':False,
            'causal_failure_mechanism_established':False,'encoder_loaded':False,'fitting_performed':False,
            'development_gradients_computed':False,'reserved_test_opened':False,'model_qualified':False,'stable_release_ready':False}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'complete':True,'summaries':summaries,'variance_ratio':report['mixed_to_uniform_head_gradient_variance_ratio'],'encoder_loaded':False},indent=2))


if __name__=='__main__':main()
