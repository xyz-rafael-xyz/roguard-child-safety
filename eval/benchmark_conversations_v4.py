"""Latency and quantization diagnostic on fixed development examples."""
import argparse, json, platform, resource, time, warnings
from pathlib import Path
import numpy as np
import torch
from roguard.conversation import ConversationScorer, render, configuration
from roguard.semantic import SemanticEncoder, head_scores
from training.conversations_v4 import generate

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    rows=[r for r in generate('dev') if r['discourse'] in ('personal','fiction','retraction')]
    rows=[r for lang in ('ro','uk') for r in [r for r in rows if r['language']==lang][:24]]
    texts=[render(r['turns']) for r in rows]
    start=time.perf_counter();float_model=ConversationScorer('semantic',args.base,'cpu');load=time.perf_counter()-start
    start=time.perf_counter();features=float_model.encoder.encode(texts);batch_seconds=time.perf_counter()-start
    fs=head_scores(features,float_model.head)
    latencies=[]
    for t in texts[:12]:
        start=time.perf_counter();float_model.encoder.encode([t]);latencies.append(time.perf_counter()-start)
    result={'platform':platform.machine(),'torch_threads':4,'cpu_load_seconds':load,'cpu_batch_rows':len(rows),
            'cpu_batch_seconds':batch_seconds,'cpu_warm_single_p50_ms':float(np.median(latencies)*1000),
            'cpu_warm_single_p95_ms':float(np.percentile(latencies,95)*1000),
            'maxrss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'includes_evidence_ablation':False,'development_diagnostic_only':True}
    try:
        start=time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');quant=SemanticEncoder(args.base,'cpu',quantize=True)
        result['int8_load_seconds']=time.perf_counter()-start
        start=time.perf_counter();qs=head_scores(quant.encode(texts),float_model.head)
        result['int8_batch_seconds']=time.perf_counter()-start
        changes=0
        for row,f,q in zip(rows,fs,qs):
            setting=configuration()['semantic'][row['language']];cut=setting['cutoff'];margin=setting['abstain_margin']
            label=lambda s: 'unknown' if abs(s-cut)<=margin else str(s>=cut)
            changes+=label(f)!=label(q)
        result['int8_max_score_difference']=float(np.max(np.abs(fs-qs)))
        result['int8_decision_changes']=int(changes)
        result['quantization_promoted']=False
    except (RuntimeError,AttributeError,NotImplementedError,ValueError,TypeError) as exc:
        result['quantization_supported']=False;result['quantization_failure_type']=type(exc).__name__
        result['quantization_promoted']=False
    fast=ConversationScorer();times=[]
    for row,text in zip(rows,texts):
        start=time.perf_counter();fast.scores([text],row['language']);times.append(time.perf_counter()-start)
    result['fast_warm_single_p50_ms']=float(np.median(times[2:])*1000)
    result['fast_warm_single_p95_ms']=float(np.percentile(times[2:],95)*1000)
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
