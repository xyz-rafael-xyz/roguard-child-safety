"""Separate model-authored diagnostic probes, never used to fit context heads."""
import argparse,hashlib,json,time
from pathlib import Path
from training.context_v7 import LABELS

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve diagnostic outputs')
    from mlx_lm import load,generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    import mlx.core as mx
    record={'synthetic_only':True,'labels':'generator-instructed, not independently verified','temperature':.3,'seed_base':710,'repetition_penalty':1.15,'runs':[]}
    definitions=['a current first-person mild need for support','a past first-person mild difficulty, clearly in the past','someone else having a mild difficulty, reported by the speaker','a fictional or quoted mild difficulty used in a literature exercise','a later correction of an earlier first-person difficulty statement: it was a quotation','uncertainty about whether a mild concern describes the speaker','an ordinary practical question without a support concern, not arithmetic']
    for specification in a.model:
        name,path=specification.split('=',1);model,tokenizer=load(path)
        for lang in ('ro','uk'):
            for index,label in enumerate(LABELS):
                mx.random.seed(710+index)
                prompt=('Create a completely fictional two-message conversation from a minor, in '+('Romanian' if lang=='ro' else 'Ukrainian')+'. The final message should establish '+definitions[index]+'. Use natural short messages, everyday wording, no names, locations, explicit harm or personal identifiers. Return ONLY JSON with exactly this shape: {"turns":[{"role":"minor","text":"..."},{"role":"minor","text":"..."}]}. Do not include the category label or English explanations. This is a synthetic research example, not a real child record.')
                formatted=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
                raw=generate(model,tokenizer,prompt=formatted,max_tokens=400,sampler=make_sampler(temp=.3),logits_processors=make_logits_processors(repetition_penalty=1.15,repetition_context_size=80),verbose=False)
                cleaned=raw.strip()
                if cleaned.startswith('```'):cleaned=cleaned.split('\n',1)[1].rsplit('```',1)[0].strip()
                row={'generator':name,'language':lang,'requested_frame':label,'raw':raw,'raw_sha256':hashlib.sha256(raw.encode()).hexdigest()}
                try:
                    value=json.loads(cleaned)
                    from roguard.context import validate
                    payload={'language':lang,'turns':value['turns']};validate(payload)
                    row['payload']=payload;row['valid_format']=True
                except (ValueError,KeyError,TypeError):row['valid_format']=False
                record['runs'].append(row);a.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(name,lang,label,row['valid_format'],flush=True)
        del model,tokenizer;mx.clear_cache()
if __name__=='__main__':main()
