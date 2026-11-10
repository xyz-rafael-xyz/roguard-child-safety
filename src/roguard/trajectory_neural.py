"""Experimental evidence bridge around the unchanged, qualified v9 components.

Short supported prefixes use exactly v9's decision path. Long prefixes use a
learned extractive view and are a new, separately evaluated task. Field sets
require their own calibration; binary qualification does not qualify fields.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .assurance import assure,digest,policy_digest
from .evidence import ONTOLOGY
from .gemma4_review import _Engine,_score,_LOCK
from .gemma4_contract import PREFIX_TOKENS,classifier_tokens
from .native_fields import FIELDS,SIZES
from .trajectory_data import prefix,input_identity
from .trajectory_memory import select
from .trajectory_sets import WORLDS,predict_set


def factors_from_logits(logits):
    values=[float(v) for v in logits]
    if len(values)!=sum(SIZES)+2 or any(not math.isfinite(v) for v in values):
        raise ValueError('Invalid neural field logits')
    factors={};offset=0
    for field,size in zip(FIELDS,SIZES):
        section=values[offset:offset+size];top=max(section)
        exp=[math.exp(v-top) for v in section];total=sum(exp)
        factors[field]=dict(zip(ONTOLOGY[field],[v/total for v in exp]));offset+=size
    probabilities=[math.prod(factors[k][w[k]] for k in ONTOLOGY) for w in WORLDS]
    total=sum(probabilities)
    return [v/total for v in probabilities]


class FieldEngine(_Engine):
    max_tokens=1024
    cache_instructions=False

    def scores(self,prefixes,spec):
        from .gemma4_attention import capture,capture_with_prefix,replay
        mx=self.mx
        sequences=[classifier_tokens(self.tokenizer,p) for p in prefixes]
        common=len(PREFIX_TOKENS);cache=None
        if any(common<len(s)<=self.max_tokens and s[:common]==PREFIX_TOKENS for s in sequences):
            cache=getattr(self,'_instruction_cache',None) if self.cache_instructions else None
            if cache is None:
                cache=self.make_cache(self.model.base)
                mx.eval(self.model.base.model(mx.array([PREFIX_TOKENS]),cache=cache))
                if self.cache_instructions:self._instruction_cache=cache
        output=[];self.last_fields=[]
        for sequence in sequences:
            if len(sequence)>self.max_tokens:
                output.append(None);self.last_fields.append(None);continue
            if cache is not None and len(sequence)>common and sequence[:common]==PREFIX_TOKENS:
                state=capture_with_prefix(self.model.base,mx.array([sequence[common:]]),cache,start=24)
            else:state=capture(self.model.base,mx.array([sequence]),start=24)
            values=replay(self.model.base,state)[:,0,:]
            logits=self.model.heads[spec['language']](values);mx.eval(logits)
            output.append(_score(logits[0],spec))
            self.last_fields.append(factors_from_logits(logits[0].tolist()))
            del state,values,logits
            mx.clear_cache()
        return output


class LongContextEngine(FieldEngine):
    """Comparator only: frozen weights beyond their qualified input envelope."""
    max_tokens=4096
    cache_instructions=True


class FullContextComparator:
    def __init__(self,components):
        self.components=components;self.active=None
        self.identity=digest({'models':{k:v.identity for k,v in components.items()},'context_limit':4096,
                              'comparator':'unqualified-full-prefix-v1'})

    def predict(self,payload):
        payload=prefix(payload,len(payload['turns'])-1);lang=payload['language'];component=self.components[lang]
        with _LOCK:
            if self.active is not None and self.active!=lang:
                self.components[self.active]._engine=None
                import gc
                gc.collect()
            self.active=lang
            if component._engine is None:component._engine=LongContextEngine(*component._args)
            if not isinstance(component._engine,LongContextEngine):raise ValueError('Dedicated comparator engine required')
            score=component._engine.scores([payload],component._spec)[0]
            from .trajectory_memory import units
            return {'predicted':None if score is None else int(score>=component._spec['threshold']),
                    'score':score,'language':lang,'focus':len(payload['turns'])-1,'input_digest':input_identity(payload),
                    'model_identity':self.identity,'probabilities':component._engine.last_fields[0] or [1/432]*432,
                    'evidence':[{k:s[k] for k in ('turn','start','end')} for s in units(payload)] if score is not None else [],
                    'qualified':False,'establishes_safety':False,'permits_external_action':False}


def view_payload(payload,view,*,preserve_complete=False):
    turns=[];last=None
    for s in view['spans']:
        if last==s['turn']:turns[-1]['text']+=' '+s['text']
        else:
            turns.append({'role':s['role'],'text':s['text']});last=s['turn']
    if preserve_complete:
        from .trajectory_memory import units
        all_units={};kept={}
        for s in units(payload):all_units.setdefault(s['turn'],set()).add((s['start'],s['end']))
        for s in view['spans']:kept.setdefault(s['turn'],set()).add((s['start'],s['end']))
        for turn,index in zip(turns,sorted(kept)):
            if kept[index]==all_units[index]:turn['text']=payload['turns'][index]['text']
    return {'language':payload['language'],'turns':turns}


def prepare_memory(payload,selector,budget,*,preserve_complete=False):
    view=select(payload,selector.relevance,budget=budget,strategy='learned',
                min_relevance=selector.spec.get('min_relevance'))
    return view,view_payload(payload,view,preserve_complete=preserve_complete)


class NeuralTrajectoryModel:
    @classmethod
    def from_profile(cls,selector,*,profile_name='v9-gemma4-20261019',budget=1200):
        from .model_install import profile,install,cache_directory
        from .language_bundle import _read_manifest
        from .field_pipeline import _local_path
        from .gemma4_review import Gemma4ReviewModel
        raw,sha=profile(profile_name)
        path,bundle_sha=install(raw,sha,cache_directory(),download=False)
        spec=_read_manifest(path,bundle_sha)
        if spec.get('format')!='roguard-language-review-bundle-v1' or set(spec.get('languages',{}))!={'ro','uk'}:
            raise ValueError('A pinned bilingual Gemma4 profile is required')
        components={}
        for lang,entry in spec['languages'].items():
            child=_local_path(path.parent,entry['path'])
            components[lang]=Gemma4ReviewModel.from_bundle(child,expected_sha256=entry['sha256'])
        return cls(components,selector,budget=budget)

    def __init__(self,components,selector,*,budget=1200,selection_mode='overflow'):
        if set(components)!={'ro','uk'}:raise ValueError('Both frozen language components are required')
        if selection_mode not in ('overflow','uniform'):raise ValueError('Unknown neural selection mode')
        self.components=dict(components);self.selector=deepcopy(selector);self.budget=budget;self.active=None
        self.selection_mode=selection_mode
        contract={'components':{k:v.identity for k,v in components.items()},'selector':selector.identity,
                  'budget':budget,'bridge':'frozen-v9-fields-extractive-v2'}
        if selection_mode=='uniform':contract['selection']='all-prefixes-preserve-complete-turns-v1'
        self.identity=digest(contract)

    def pipeline_digest(self):
        return digest({'model':self.identity,'policy':policy_digest(),'field_distribution':'independent-six-fields-v1'})

    def predict(self,payload,*,focus=None,calibration=None,plan=False):
        payload=prefix(payload,len(payload['turns'])-1 if focus is None else focus)
        lang=payload['language'];component=self.components[lang]
        with _LOCK:
            if self.active is not None and self.active!=lang:
                self.components[self.active]._engine=None
                import gc
                gc.collect()
            self.active=lang
            if component._engine is None:component._engine=FieldEngine(*component._args)
            elif not isinstance(component._engine,FieldEngine):raise ValueError('Dedicated neural evidence engine required')
            raw=None;view=None;inference=payload
            if self.selection_mode=='overflow' and len(payload['turns'])<=12:
                try:
                    raw=component.predict(payload)
                except ValueError:
                    # Validate the original prefix independently. Runtime errors
                    # must propagate rather than being hidden by retrieval.
                    from .interpreter import prepare_input
                    try:prepare_input(payload)
                    except ValueError:pass
                    else:raise
            if raw is None or raw['predicted'] is None:
                view,inference=prepare_memory(payload,self.selector,self.budget,
                                              preserve_complete=self.selection_mode=='uniform')
                if view['mandatory_overflow'] or not inference['turns'] or len(inference['turns'])>12:
                    raw={'predicted':None,'score':None,'reason':'EVIDENCE_VIEW_EXCEEDS_V9_CONTRACT'}
                    probabilities=[1/432]*432
                else:
                    raw=component.predict(inference)
                    probabilities=component._engine.last_fields[0] or [1/432]*432
            else:probabilities=component._engine.last_fields[0] or [1/432]*432
            if view is None:
                from .trajectory_memory import units
                evidence=[{k:s[k] for k in ('turn','start','end')} for s in units(payload)]
            else:evidence=[{k:s[k] for k in ('turn','start','end')} for s in view['spans']]
            result={**raw,'format':'roguard-neural-trajectory-v10-research1','focus':len(payload['turns'])-1,
                    'language':lang,'input_digest':input_identity(payload),'model_identity':self.identity,
                    'pipeline_digest':self.pipeline_digest(),'inference_input_digest':input_identity(inference) if inference['turns'] else None,
                    'probabilities':probabilities,'evidence':evidence,'used_memory':view is not None,
                    'qualified':False,'establishes_safety':False,'permits_external_action':False}
            if calibration is not None:
                sets=predict_set(probabilities,calibration,language=lang,pipeline_digest=self.pipeline_digest(),supported=raw['predicted'] is not None)
                result['interpretation_set']=sets
                if plan:
                    result['assurance']=assure({'language':lang,'context_digest':result['input_digest'],
                        'hypotheses':sets['hypotheses'],'model_digests':[self.identity],'review_floor':raw['predicted']==1})
            return result
