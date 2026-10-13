"""Run cached actual generative models on identical fictional v5 scenarios."""
import argparse,hashlib,json,time,subprocess
from pathlib import Path
from roguard.targets import SYSTEM,parse_reply
from roguard.interaction_lab import run_scenario
from roguard.lab_scenarios import generate_suite
from roguard.lab_metrics import summarize_runs

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve comparison')
    from mlx_lm import load,generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    scenarios=generate_suite(seed=20261026,repetitions=1)
    record=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seed=20261026,temperature=.3,repeats=2,synthetic_only=True,host_minimum_proposal_visible=True,models={})
    for spec in a.model:
        name,path=spec.split('=',1);root=Path(path)
        files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.iterdir()) if f.is_file() and (f.suffix in ('.json','.safetensors') or f.name=='tokenizer.model')}
        model,tokenizer=load(root);runs=[];times=[];invalid=[]
        for repeat in range(2):
            mx.random.seed(20261026+repeat)
            for scenario in scenarios:
                def target(observation):
                    prompt=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(observation,ensure_ascii=False)}],tokenize=False,add_generation_prompt=True)
                    started=time.perf_counter();raw=generate(model,tokenizer,prompt=prompt,max_tokens=192,sampler=make_sampler(temp=.3),verbose=False);times.append(time.perf_counter()-started)
                    try:return parse_reply(raw)
                    except ValueError:
                        invalid.append(hashlib.sha256(raw.encode()).hexdigest());return {'malformed':True}
                run=run_scenario(scenario,lambda:target,enforce=True);runs.append(run)
                print(name,repeat,scenario['case_id'],'requests',len(run['requests']),flush=True)
        record['models'][name]=dict(file_hashes=files,metrics=summarize_runs(runs),by_language={l:summarize_runs([r for r in runs if r['language']==l]) for l in ('ro','uk')},inference_seconds=sum(times),calls=len(times),invalid_output_hashes=invalid,runs=runs)
        a.output.write_text(json.dumps(record,indent=2));del model,tokenizer;mx.clear_cache()
if __name__=='__main__':main()
