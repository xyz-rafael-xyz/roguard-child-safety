"""Bounded conversational evidence proposals. Predictions never grant permission."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import unicodedata

LABELS = ('SELF_CURRENT','SELF_PAST','OTHER_PERSON','QUOTED','CORRECTION','UNCERTAIN','ORDINARY')
QUESTIONS = {
    'SELF_CURRENT':'A reviewer should check whether this describes a current personal concern.',
    'SELF_PAST':'A reviewer should check whether the past event still affects the speaker.',
    'OTHER_PERSON':'A reviewer should check who the statement concerns; the speaker may be reporting another person.',
    'QUOTED':'A reviewer should check the quotation or fictional context before interpreting it as personal.',
    'CORRECTION':'A reviewer should reconcile this correction with earlier statements; no earlier signal is erased.',
    'UNCERTAIN':'A reviewer should examine the ambiguity without assuming the statement is true or false.',
    'ORDINARY':'No contextual concern was identified by this experimental frame classifier.',
}

def render(turns):
    return '\n'.join(('FOCUS ' if i==len(turns)-1 else 'CONTEXT ')+t['role']+': '+unicodedata.normalize('NFC',t['text']) for i,t in enumerate(turns))

def validate(payload):
    if not isinstance(payload,dict) or set(payload)!={'language','turns'}:
        raise ValueError('Expected language and turns')
    if payload['language'] not in ('ro','uk'):
        raise ValueError('Language must be ro or uk')
    turns=payload['turns']
    if not isinstance(turns,list) or not 1<=len(turns)<=12:
        raise ValueError('Expected 1 to 12 turns')
    for t in turns:
        if not isinstance(t,dict) or set(t)!={'role','text'} or t['role'] not in ('minor','assistant','adult','other','unknown'):
            raise ValueError('Invalid turn')
        if not isinstance(t['text'],str) or not t['text'].strip() or len(t['text'])>2000:
            raise ValueError('Each turn needs 1 to 2000 characters')
    if sum(len(t['text']) for t in turns)>6000:
        raise ValueError('Conversation exceeds 6000 characters')
    return turns

class ContextModel:
    """Experimental frame and mention heads, fitted on fictional contexts."""
    def __init__(self,backend='joint',base_path=None,device='cpu',artifact=None):
        if backend not in ('character','semantic','joint'):raise ValueError('Unknown context backend')
        data=Path(artifact).read_bytes() if artifact else resources.read_binary('roguard','context-'+backend+'.json.gz')
        self.identity=hashlib.sha256(data).hexdigest()
        self.spec=json.loads(gzip.decompress(data));self.backend=backend
        if self.spec.get('format')!='roguard-context-v1' or self.spec.get('backend')!=backend:raise ValueError('Context model format mismatch')
        if set(self.spec.get('heads',{}))!={'frame','signal'}:raise ValueError('Missing context heads')
        dimension=len(self.spec.get('features',[])) if backend=='character' else 1536
        if not 1<=dimension<=50000:raise ValueError('Invalid feature count')
        for task,head in self.spec['heads'].items():
            expected=set(LABELS) if task=='frame' else {0,1}
            if set(head['classes'])!=expected or len(head['classes'])!=len(expected):raise ValueError('Invalid classes')
            count=len(LABELS) if task=='frame' else 1
            if len(head['weights'])!=count or len(head['intercept'])!=count:raise ValueError('Invalid head shape')
            if any(len(row)!=dimension or any(not math.isfinite(v) for v in row) for row in head['weights']):raise ValueError('Invalid weights')
            if any(not math.isfinite(v) for v in head['intercept']):raise ValueError('Invalid intercept')
            if set(head['quantiles'])!={'ro','uk'} or any(len(q)!=len(expected) or any(not math.isfinite(v) or not 0<=v<=1 for v in q) for q in head['quantiles'].values()):raise ValueError('Invalid calibration')
        self.encoder=None
        if backend in ('semantic','joint'):
            from .semantic import SemanticEncoder
            if base_path is None:
                from .model_setup import locate_model
                base_path=locate_model()
            self.encoder=SemanticEncoder(base_path,device)
            if backend=='joint':
                import torch
                from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
                adapter=self.spec['lora']
                if adapter['rank']!=4 or adapter['alpha']!=8:raise ValueError('Unsupported context adaptation')
                model=get_peft_model(self.encoder.model,LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,target_modules=['Wqkv']))
                expected=get_peft_model_state_dict(model)
                if set(expected)!=set(adapter['tensors']):raise ValueError('Context adapter keys differ')
                tensors={}
                for name,value in adapter['tensors'].items():
                    tensor=torch.tensor(value,dtype=torch.float32)
                    if tensor.shape!=expected[name].shape or not torch.isfinite(tensor).all():raise ValueError('Invalid context adapter tensor')
                    tensors[name]=tensor
                set_peft_model_state_dict(model,tensors);self.encoder.model=model.to(device).eval()
        else:
            self.features={name:(i,idf) for i,(name,idf) in enumerate(self.spec['features'])}

    def _features(self,text):
        counts=Counter(text.lower()[i:i+n] for n in range(2,6) for i in range(len(text)-n+1))
        values={self.features[s][0]:(1+math.log(n))*self.features[s][1] for s,n in counts.items() if s in self.features}
        norm=math.sqrt(sum(v*v for v in values.values()))
        return {i:v/norm for i,v in values.items()} if norm else {}

    def predict(self,texts,language):
        if language not in ('ro','uk'):raise ValueError('Unsupported language')
        if not isinstance(texts,list) or len(texts)>10000:raise ValueError('Invalid context batch')
        results=[]
        for text in texts:
            if not isinstance(text,str) or len(text)>7000:raise ValueError('Context model input too large')
            if language=='uk' and not any('\u0400'<=c<='\u04ff' for c in text):
                results.append({'supported':False,'reason':'UKRAINIAN_SCRIPT_REQUIRED','heads':{}});continue
            try:
                if self.encoder:
                    import numpy as np
                    features=np.concatenate(self.encoder.encode([text, 'FOCUS '+text.rsplit('\nFOCUS ',1)[-1].removeprefix('FOCUS ')]))
                else:
                    features=self._features(text)
            except ValueError:
                results.append({'supported':False,'reason':'SEMANTIC_CONTEXT_LIMIT','heads':{}});continue
            heads={}
            for task,head in self.spec['heads'].items():
                logits=[float(b+sum(features[i]*w[i] for i in features)) if isinstance(features,dict) else float(b+features@w) for w,b in zip(head['weights'],head['intercept'])]
                if len(logits)==1:logits=[0,logits[0]]
                maximum=max(logits);exps=[math.exp(v-maximum) for v in logits];total=sum(exps);probs=[v/total for v in exps]
                labels=head['classes'];prediction_set=[label for label,p,q in zip(labels,probs,head['quantiles'][language]) if 1-p<=q+1e-12]
                heads[task]={'prediction_set':prediction_set,'label':prediction_set[0] if len(prediction_set)==1 and prediction_set[0]==labels[max(range(len(probs)),key=probs.__getitem__)] else None,'argmax':labels[max(range(len(probs)),key=probs.__getitem__)]}
            results.append({'supported':True,'heads':heads})
        return results

def analyze_context(payload,model=None):
    turns=validate(payload);model=model or ContextModel()
    # All preceding turns are retained. No silent window selection or truncation.
    indices=[i for i,t in enumerate(turns) if t['role']=='minor']
    predictions=model.predict([render(turns[:i+1]) for i in indices],payload['language'])
    evidence=[];review_refs=[];unresolved=[]
    for i,prediction in zip(indices,predictions):
        frame=prediction['heads'].get('frame',{});signal=prediction['heads'].get('signal',{})
        label=frame.get('label');mentioned=signal.get('label')
        uncertain=not prediction['supported'] or mentioned is None or (mentioned==1 and (label is None or label in ('CORRECTION','UNCERTAIN')))
        needs_review=uncertain or (mentioned==1 and label in ('SELF_CURRENT','SELF_PAST','OTHER_PERSON'))
        if needs_review:review_refs.append(i)
        if uncertain:unresolved.append(i)
        evidence.append({'turn_index':i,'context_turn_indices':list(range(i)),
            'candidate_reference':{'turn_index':i,'start':0,'end':len(turns[i]['text']),'offset_unit':'unicode_code_points'},
            'frame':label,'frame_set':frame.get('prediction_set',[]),'mention':mentioned,
            'mention_set':signal.get('prediction_set',[]),'supported':prediction['supported'],
            'frame_undecided':label is None,'review_recommended':needs_review,'reason':prediction.get('reason','MODEL_PROPOSAL'),
            'review_question':QUESTIONS.get(label,'The context is unresolved; a reviewer should inspect the conversation.'),
            'verified':False,'causal_attribution':False})
    return {'schema_version':'7.0-experimental','language':payload['language'],'backend':model.backend,
        'model_sha256':model.identity,'evidence':evidence,'review_turn_indices':review_refs,'unresolved_turn_indices':unresolved,
        'decision':'REVIEW' if review_refs else 'NO_REVIEW_SIGNAL' if indices else 'INSUFFICIENT_EVIDENCE',
        'earlier_signals_retained':True,'authority_from_model':False,'external_action_taken':False,
        'real_world_accuracy_established':False,'text_in_report':False,
        'limits':['Synthetic training and calibration','Whole-turn candidate references, not verified span extraction',
                  'Corrections never automatically dismiss earlier concerns','No claim that an unflagged conversation is safe']}

def compare_context(before,after):
    """Compare reports only from the same model and language; no text is retained."""
    if before.get('model_sha256')!=after.get('model_sha256') or before.get('language')!=after.get('language'):
        raise ValueError('Comparison needs matching model and language')
    old={e['turn_index']:e for e in before['evidence']}
    return {'added_turn_indices':[e['turn_index'] for e in after['evidence'] if e['turn_index'] not in old],
            'changed_turn_indices':[e['turn_index'] for e in after['evidence'] if e['turn_index'] in old and e!=old[e['turn_index']]],
            'new_review_turn_indices':sorted(set(after['review_turn_indices'])-set(before['review_turn_indices'])),
            'external_action_taken':False}

def main():
    p=argparse.ArgumentParser(description='Experimental local conversation evidence review')
    p.add_argument('input',type=Path);p.add_argument('--backend',choices=['character','semantic','joint'],default='joint');p.add_argument('--model-path',type=Path);p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    a=p.parse_args()
    from .cli import _strict_json_loads
    try:
        if a.input.stat().st_size>64000:raise ValueError('Input too large')
        result=analyze_context(_strict_json_loads(a.input.read_text()),ContextModel(a.backend,a.model_path,a.device))
    except (ValueError,OSError,ImportError):p.error('Invalid input or missing model/dependencies; see docs/CONTEXT.md')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
