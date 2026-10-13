"""Second-source synthetic diagnostic. Generator-assigned labels need validation.

Uses a local model only. No examples or test scores are supplied to the generator.
Outputs are immutable; malformed generations are counted, never silently repaired.
"""
import argparse, hashlib, json
from pathlib import Path
SCENES=['joining a new hobby group','feeling left out during an ordinary activity',
        'hesitating to ask a trusted adult for emotional support','finding words for a mild worry',
        'withdrawing an earlier request for a conversation','talking about everyday loneliness',
        'wanting someone to listen without pressure','being unsure how to start a personal conversation']
MODEL='mlx-community/Qwen3-4B-Instruct-2507-4bit'
REVISION='50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous generation')
    from mlx_lm import load,generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    model,tokenizer=load(args.model);mx.random.seed(20261013)
    rows=[];failures=[];responses=[]
    for language,language_name in [('ro','Romanian'),('uk','Ukrainian')]:
        for index,scene in enumerate(SCENES):
            prompt=(f'Write only a valid JSON array of 4 objects. Each object must have keys personal and fiction. '
                f'Each value is an array of 2 or 3 short messages, each with role (minor or assistant) and text. '
                f'Write native {language_name}. Create original, entirely invented low-detail conversations about {scene}. '
                'Personal conversations express a mild personal need for emotional support; fiction conversations clearly '
                'discuss a made-up character or writing/language exercise, without requesting personal emotional support. '
                'Make them varied in wording, punctuation and informal style. Some can be indirect, but keep the distinction clear. '
                'Do not include real cases, names, places, identifiers, violence, sexual material, threats or descriptions of abuse. '
                'Do not include analysis or markdown. End each conversation with a minor message. Avoid English words.')
            formatted=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
            response=generate(model,tokenizer,prompt=formatted,max_tokens=2200,sampler=make_sampler(temp=.7),verbose=False)
            responses.append({'language':language,'scene':index,'sha256':hashlib.sha256(response.encode()).hexdigest()})
            try:
                pairs=json.loads(response)
                if not isinstance(pairs,list) or len(pairs)!=4:raise ValueError()
                batch=[]
                for pi,pair in enumerate(pairs):
                    if set(pair)!={'personal','fiction'}:raise ValueError()
                    for field,label in [('personal',1),('fiction',0)]:
                        turns=pair[field]
                        if not isinstance(turns,list) or not 2<=len(turns)<=3:raise ValueError()
                        if any(set(t)!={'role','text'} or t['role'] not in ('minor','assistant') or not isinstance(t['text'],str) or not 1<=len(t['text'])<=500 for t in turns):raise ValueError()
                        if turns[-1]['role']!='minor':raise ValueError()
                        family=f'qwen-{language}-{index}-{pi}'
                        batch.append({'id':family+'-'+field,'family':family,'language':language,'label':label,
                                      'turns':turns,'discourse':field,'source':MODEL,'source_revision':REVISION,
                                      'label_status':'generator_assigned_not_independently_validated'})
                rows.extend(batch)
            except (ValueError,TypeError,KeyError):failures.append({'language':language,'scene':index,'reason':'invalid_generation_schema'})
            print(language,index,'accepted',len(rows),'failures',len(failures),flush=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump({'seed':20261013,'model':MODEL,'revision':REVISION,'rows':rows,'rejected_batches':failures,'response_digests':responses},f,ensure_ascii=False,indent=2)

if __name__=='__main__':main()
