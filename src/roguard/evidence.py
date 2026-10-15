"""Experimental factorized evidence proposals and conservative local cascade.

Models propose interpretations; they cannot establish facts or permissions.
Paragraph references are candidate evidence units, not verified predicate spans.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import re
from .context import validate

ONTOLOGY = {
    'subject': ('SELF','OTHER','FICTIONAL'),
    'time': ('CURRENT','PAST','HYPOTHETICAL'),
    'assertion': ('AFFIRMED','NEGATED','UNCERTAIN','RETRACTED'),
    'source': ('DIRECT','REPORTED','FICTION'),
    'relation': ('NEW','RETRACTS'),
    'concern': ('0','1'),
}


def units(payload):
    """Preserve original code-point offsets, including combining characters."""
    turns=validate(payload);out=[]
    for i,turn in enumerate(turns):
        if turn['role']!='minor':continue
        for m in re.finditer(r'[^\n]+',turn['text']):
            text=m.group();left=len(text)-len(text.lstrip());right=len(text.rstrip())
            if right>left:
                out.append({'id':f't{i}_s{m.start()+left}', 'turn_index':i,
                            'start':m.start()+left,'end':m.start()+right,
                            'text':text[left:right]})
    if len(out)>48:raise ValueError('At most 48 nonempty minor paragraphs are supported')
    return out


class EvidenceModel:
    def __init__(self, backend='character', artifact=None, base_path=None, device='cpu'):
        if backend not in ('character','semantic'):raise ValueError('Unknown evidence backend')
        data=Path(artifact).read_bytes() if artifact else resources.read_binary('roguard',f'evidence-{backend}.json.gz')
        self.identity=hashlib.sha256(data).hexdigest();self.backend=backend
        self.spec=json.loads(gzip.decompress(data))
        if self.spec.get('format')!='roguard-evidence-v1' or self.spec.get('backend')!=backend:raise ValueError('Invalid evidence artifact')
        if set(self.spec.get('heads',{}))!=set(ONTOLOGY):raise ValueError('Invalid evidence ontology')
        features=self.spec.get('features',[]);dimension=len(features) if backend=='character' else (1536 if self.spec.get('representation')=='context_plus_focus' else 768)
        if not 1<=dimension<=40000:raise ValueError('Invalid feature dimension')
        for task,head in self.spec['heads'].items():
            if set(head['classes'])!=set(ONTOLOGY[task]) or len(head['classes'])!=len(ONTOLOGY[task]):raise ValueError('Invalid classes')
            n=len(head['classes']);count=1 if n==2 else n
            if len(head['weights'])!=count or len(head['intercept'])!=count:raise ValueError('Invalid head dimensions')
            if any(len(w)!=dimension or any(not math.isfinite(v) for v in w) for w in head['weights']):raise ValueError('Invalid head weights')
            if any(not math.isfinite(v) for v in head['intercept']):raise ValueError('Invalid head bias')
            if set(head['quantiles'])!={'ro','uk'} or any(len(q)!=n or any(not math.isfinite(v) or not 0<=v<=1 for v in q) for q in head['quantiles'].values()):raise ValueError('Invalid calibration')
        self.encoder=None
        if backend=='character':
            if any(not isinstance(n,str) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for n,v in features):raise ValueError('Invalid vocabulary')
            self.features={n:(i,v) for i,(n,v) in enumerate(features)}
            if len(self.features)!=dimension:raise ValueError('Duplicate features')
        else:
            from .semantic import SemanticEncoder
            if base_path is None:
                from .model_setup import locate_model
                base_path=locate_model()
            self.encoder=SemanticEncoder(base_path,device)
            if 'lora' in self.spec:
                import torch
                from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
                adapter=self.spec['lora']
                if adapter.get('rank')!=4 or adapter.get('alpha')!=8:raise ValueError('Invalid evidence adaptation')
                model=get_peft_model(self.encoder.model,LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,target_modules=['Wqkv']))
                expected=get_peft_model_state_dict(model)
                if set(expected)!=set(adapter['tensors']):raise ValueError('Evidence adapter keys differ')
                tensors={}
                for name,value in adapter['tensors'].items():
                    tensor=torch.tensor(value,dtype=torch.float32)
                    if tensor.shape!=expected[name].shape or not torch.isfinite(tensor).all():raise ValueError('Invalid adapter tensor')
                    tensors[name]=tensor
                set_peft_model_state_dict(model,tensors);self.encoder.model=model.to(device).eval()

    def _features(self,text):
        # sklearn's char analyzer collapses runs of whitespace before ngrams.
        text=re.sub(r'\s\s+',' ',text.lower())
        counts=Counter(text[i:i+n] for n in range(2,6) for i in range(len(text)-n+1))
        values={self.features[s][0]:(1+math.log(n))*self.features[s][1] for s,n in counts.items() if s in self.features}
        norm=math.sqrt(sum(v*v for v in values.values()))
        return {i:v/norm for i,v in values.items()} if norm else {}

    def predict(self,texts,language):
        if language not in ('ro','uk') or not isinstance(texts,list) or len(texts)>10000:raise ValueError('Invalid evidence batch')
        result=[]
        for item in texts:
            text=item['text'] if isinstance(item,dict) else item
            context=item.get('context',text) if isinstance(item,dict) else text
            if not isinstance(context,str) or len(context)>7000:raise ValueError('Invalid evidence context')
            if not isinstance(text,str) or not text.strip() or len(text)>2000:raise ValueError('Invalid evidence unit')
            if language=='uk' and not any('\u0400'<=c<='\u04ff' for c in text):
                result.append({'supported':False,'reason':'UKRAINIAN_SCRIPT_REQUIRED','attributes':{}});continue
            try:
                if self.encoder and self.spec.get('representation')=='context_plus_focus':
                    import numpy as np
                    features=np.concatenate(self.encoder.encode([context,text]))
                else:features=self.encoder.encode([text])[0] if self.encoder else self._features(text)
            except ValueError:
                result.append({'supported':False,'reason':'SEMANTIC_CONTEXT_LIMIT','attributes':{}});continue
            heads={}
            for task,h in self.spec['heads'].items():
                logits=[float(b+sum(features[i]*w[i] for i in features)) if isinstance(features,dict) else float(b+features@w) for w,b in zip(h['weights'],h['intercept'])]
                if len(logits)==1:logits=[0,logits[0]]
                maximum=max(logits);exp=[math.exp(v-maximum) for v in logits];total=sum(exp);p=[v/total for v in exp]
                winner=h['classes'][max(range(len(p)),key=p.__getitem__)]
                labels=[c for c,prob,q in zip(h['classes'],p,h['quantiles'][language]) if 1-prob<=q+1e-12]
                heads[task]={'label':labels[0] if len(labels)==1 and labels[0]==winner else None,
                             'prediction_set':labels,'argmax':winner}
            result.append({'supported':True,'reason':'MODEL_PROPOSAL','attributes':heads})
        return result


class EvidenceCascade:
    """Optional challenger, invoked only for unresolved primary units.

    The challenger is an additional model, not an oracle. A decided conflict
    becomes unresolved; a singleton can resolve an overlapping primary set.
    """
    def __init__(self,primary=None,challenger=None):
        self.primary=primary or EvidenceModel();self.challenger=challenger
        self.identity=self.primary.identity+(':'+challenger.identity if challenger else '')
    def predict(self,texts,language):
        predictions=deepcopy(self.primary.predict(texts,language))
        for text,p in zip(texts,predictions):
            p['challenger_used']=False;p['disagreements']=[]
            if not p['supported'] or not self.challenger or all(h['label'] is not None for h in p['attributes'].values()):continue
            q=self.challenger.predict([text],language)[0];p['challenger_used']=True
            if not q['supported']:p['challenger_reason']=q['reason'];continue
            for name,head in p['attributes'].items():
                other=q['attributes'][name];a=head['label'];b=other['label']
                head['primary']=deepcopy(head);head['challenger']=deepcopy(other)
                if a is not None and b is not None and a!=b:
                    head['label']=None;head['prediction_set']=sorted(set(head['prediction_set']+other['prediction_set']));p['disagreements'].append(name)
                elif a is None and b is not None and b in head['prediction_set'] and (head.get('argmax') not in ONTOLOGY[name] or head['argmax'] in head['prediction_set']):
                    head['label']=b;head['prediction_set']=[b]
            # Cascade sets are not the primary model's calibrated sets.
        return predictions


class GuardedEvidence:
    """Preserve released context review signals alongside experimental attributes.

    This is a conservative union of review proposals, not calibrated fusion.
    It cannot reduce the released workflow's review workload on the same inputs.
    """
    def __init__(self,primary,context_model):
        self.primary=primary;self.context_model=context_model
        self.identity=primary.identity+':review-floor:'+context_model.identity
    def predict(self,texts,language):
        from .context import analyze_context
        outputs=deepcopy(self.primary.predict(texts,language))
        contexts=[t.get('guard_context',t.get('context',t['text'])) if isinstance(t,dict) else t for t in texts]
        unique=list(dict.fromkeys(contexts));predictions=self.context_model.predict(unique,language)
        if len(predictions)!=len(unique):raise ValueError('Incomplete review-floor output')
        by_context=dict(zip(unique,predictions))
        class Replay:
            backend='joint';identity=self.context_model.identity
            def __init__(self,p):self.p=p
            def predict(self,texts,language):return [self.p for _ in texts]
        for item,context,output in zip(texts,contexts,outputs):
            text=item.get('guard_text',item['text']) if isinstance(item,dict) else item
            review=analyze_context({'language':language,'turns':[{'role':'minor','text':text}]},Replay(by_context[context]))['evidence'][0]
            output['review_floor']={'review_recommended':review['review_recommended'],'review_kind':review['review_kind'],'frame':review['frame'],'mention':review['mention'],'model_identity':self.context_model.identity}
        return outputs


def load_evidence_model(backend='character',base_path=None,device='cpu'):
    if backend=='guarded':
        from .context import ContextModel
        return GuardedEvidence(EvidenceModel('semantic',base_path=base_path,device=device),ContextModel('joint',base_path=base_path,device=device))
    return EvidenceModel(backend,base_path=base_path,device=device)


def _complete_action(a):
    if a['concern']=='0':return 'CONTINUE'
    if ((a['subject']=='FICTIONAL') != (a['source']=='FICTION')):return 'REVIEW_AND_CONTINUE'
    if a['assertion']=='RETRACTED' or a['relation']=='RETRACTS':return 'REVIEW_AND_CONTINUE'
    if a['source']=='FICTION' or a['assertion']=='NEGATED':return 'CONTINUE'
    if a['assertion']=='UNCERTAIN' or a['time']=='HYPOTHETICAL':return 'REVIEW_AND_CONTINUE'
    return 'SUPPORT_AND_REVIEW'


def decision_envelope(attributes):
    """Enumerate the small attribute completion space, never resolve language.

    An action can be stable even when an irrelevant attribute is undecided.
    This is exact only relative to the supplied hypothesis sets and policy.
    """
    import itertools
    names=list(ONTOLOGY);choices=[]
    for name in names:
        h=attributes.get(name,{})
        label=h.get('label')
        values=[label] if label is not None else list(h.get('prediction_set') or ONTOLOGY[name])
        # A class-conditional set can exclude argmax while retaining a lower-score
        # label. This disagreement must not silently become a certain action.
        if label is None and h.get('argmax') in ONTOLOGY[name] and h['argmax'] not in values:
            values.append(h['argmax'])
        if not isinstance(values,(list,tuple)) or not values or any(v not in ONTOLOGY[name] for v in values):raise ValueError('Invalid attribute hypotheses')
        choices.append(list(dict.fromkeys(values)))
    worlds=[];witnesses={}
    for values in itertools.product(*choices):
        world=dict(zip(names,values));action=_complete_action(world)
        worlds.append((world,action));witnesses.setdefault(action,world)
    questions=[]
    for name,values in zip(names,choices):
        if len(values)<2:continue
        worst=max(len({action for world,action in worlds if world[name]==value}) for value in values)
        questions.append({'attribute':name,'worst_remaining_actions':worst})
    questions.sort(key=lambda q:(q['worst_remaining_actions'],q['attribute']))
    actions=sorted(witnesses)
    return {'possible_actions':actions,'stable':len(actions)==1,'witnesses':witnesses,
            'hypothesis_count':len(worlds),'reviewer_questions':questions,
            'effective_hypotheses':dict(zip(names,choices)),
            'scope':'exact_given_supplied_hypotheses_and_policy','real_world_certainty':False}


def disposition(attributes):
    envelope=decision_envelope(attributes)
    if not envelope['stable']:return 'REVIEW_AND_CONTINUE','UNCERTAIN_INTERPRETATION'
    action=envelope['possible_actions'][0]
    reason='NO_CONCERN_SIGNAL' if attributes.get('concern',{}).get('label')=='0' else 'STABLE_ACTION_ACROSS_HYPOTHESES' if action=='CONTINUE' else 'CONCERN_PROPOSAL' if action=='SUPPORT_AND_REVIEW' else 'REVISION_OR_STATEMENT_REQUIRES_REVIEW'
    return action,reason


def analyze_evidence(payload,model=None):
    spans=units(payload);model=model or EvidenceCascade()
    from .context import render
    inputs=[]
    for s in spans:
        prefix=deepcopy(payload['turns'][:s['turn_index']+1]);prefix[-1]['text']=prefix[-1]['text'][:s['end']]
        inputs.append({'text':s['text'],'context':render(prefix),'guard_context':render(payload['turns'][:s['turn_index']+1]),'guard_text':payload['turns'][s['turn_index']]['text']})
    predictions=model.predict(inputs,payload['language']);out=[];claim_keys=[];edges=[]
    if len(predictions)!=len(spans):raise ValueError('Incomplete model output')
    for span,p in zip(spans,predictions):
        action,reason=disposition(p['attributes']);attribute_action=action
        floor=p.get('review_floor')
        if floor and floor['review_recommended'] and action=='CONTINUE':
            action='REVIEW_AND_CONTINUE';reason='RELEASED_MODEL_REVIEW_RETAINED'
        reference={k:v for k,v in span.items() if k!='text'}
        reference['offset_unit']='unicode_code_points'
        links=[]
        quoted=re.search('«([^«»]+)»',span['text'])
        claim=quoted.group(1) if quoted else span['text']
        # Exact repeated wording can propose an edge; it never establishes identity.
        reference['candidate_claim_start']=span['start']+(quoted.start(1) if quoted else 0)
        reference['candidate_claim_end']=span['start']+(quoted.end(1) if quoted else len(span['text']))
        normalized=' '.join(claim.casefold().split())
        current={k:h.get('label') for k,h in p['attributes'].items()}
        for previous,key in zip(out,claim_keys):
            earlier={k:h.get('label') for k,h in previous['attributes'].items()}
            matching=all(current.get(k) is not None and current.get(k)==earlier.get(k) for k in ('subject','time','source'))
            if normalized==key and matching:
                relation='POSSIBLE_RETRACTION' if current.get('assertion')=='RETRACTED' else 'POSSIBLE_CONTRADICTION' if {current.get('assertion'),earlier.get('assertion')}=={'AFFIRMED','NEGATED'} else 'REPEATED_ASSERTION' if current.get('assertion')==earlier.get('assertion') else 'RELATED_WORDING'
                edges.append({'from':span['id'],'to':previous['id'],'relation':relation,'verified':False,'basis':'matching_wording_and_predicted_scope'})
        claim_keys.append(normalized)
        if p['attributes'].get('relation',{}).get('label')=='RETRACTS':
            # Multiple plausible targets stay explicit. This never supersedes them.
            subject=p['attributes'].get('subject',{}).get('label')
            links=[e['id'] for e in out if subject is not None and e['attributes'].get('subject',{}).get('label')==subject]
        out.append(dict(id=span['id'],reference=reference,attributes=p['attributes'],
                        supported=p['supported'],reason=reason,model_reason=p['reason'],
                        action_envelope=decision_envelope(p['attributes']),
                        proposed_action=action,attribute_action=attribute_action,review_floor=floor,candidate_revision_targets=links,
                        challenger_used=p.get('challenger_used',False),disagreements=p.get('disagreements',[]),
                        verified=False))
    return {'schema_version':'8.0-experimental','language':payload['language'],
            'model_identity':model.identity,'evidence':out,'candidate_edges':edges,
            'review_ids':[e['id'] for e in out if e['proposed_action']!='CONTINUE'],
            'external_action_taken':False,'authority_from_model':False,'text_in_report':False,
            'real_world_accuracy_established':False,'cascade_coverage_guaranteed':False,
            'limits':['Synthetic controlled training','Paragraph-level proposals, not verified spans',
                      'Character model uses paragraphs; optional contextual semantic model uses bounded prefixes',
                      'Source and subject were correlated in training','No inference that unflagged content is safe']}


class EvidenceState:
    """Append-only report history with explicit host-reviewed supersession.

    Reviewer authentication belongs to the host. This API is not a model tool.
    Supersession changes the active view and never deletes historical evidence.
    """
    def __init__(self):self._report=None;self._revisions=[];self._closed=False
    def update(self,report):
        if self._closed:raise ValueError('Closed evidence state')
        if report.get('schema_version')!='8.0-experimental':raise ValueError('Invalid evidence report')
        ids=[e['id'] for e in report['evidence']]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate evidence IDs')
        expected_reviews=[e['id'] for e in report['evidence'] if e.get('proposed_action')!='CONTINUE']
        if report.get('review_ids')!=expected_reviews:raise ValueError('Review index differs from evidence')
        if self._report:
            if any(report.get(k)!=self._report.get(k) for k in ('language','model_identity')):raise ValueError('Cannot mix models or languages')
            old={e['id']:e for e in self._report['evidence']};new={e['id']:e for e in report['evidence']}
            if any(new.get(k)!=v for k,v in old.items()):raise ValueError('Evidence history must be append-only')
        self._report=deepcopy(report)
        return self.snapshot()
    def confirm_revision(self,new_id,old_id,*,reviewer_id):
        from .flow import identifier
        if self._closed or not self._report:raise ValueError('No active evidence state')
        identifier(reviewer_id,'reviewer')
        es={e['id']:e for e in self._report['evidence']}
        if new_id not in es or old_id not in es or es[new_id]['reference']['turn_index']<=es[old_id]['reference']['turn_index']:raise ValueError('Revision must follow its target')
        if any(r['old_id']==old_id for r in self._revisions):raise ValueError('Already superseded')
        self._revisions.append(dict(new_id=new_id,old_id=old_id,reviewer_id=reviewer_id,authority='HOST_REVIEW'))
        return self.snapshot()
    def snapshot(self):
        superseded={r['old_id'] for r in self._revisions}
        return {'report':deepcopy(self._report),'revisions':deepcopy(self._revisions),
                'active_review_ids':[i for i in (self._report or {}).get('review_ids',[]) if i not in superseded],
                'external_action_taken':False}
    def close(self):self._report=None;self._revisions.clear();self._closed=True


def plan_response(report,candidates,ledger,*,purpose,recipient,operation='use'):
    """Rank host-declared response alternatives after live permission checks.

    No content generation or execution. Costs/coverage are declarations; this
    does not certify that response prose actually fulfils those declarations.
    """
    from .flow import identifier, identifiers
    required={'task'}
    if any(e['proposed_action']=='SUPPORT_AND_REVIEW' for e in report['evidence']):required.add('support')
    if report['review_ids']:required.add('review')
    if not isinstance(candidates,list) or not 1<=len(candidates)<=32:raise ValueError('Invalid response candidates')
    considered=[];seen=set()
    for c in candidates:
        if not isinstance(c,dict) or set(c)!={'id','artifacts','covers','disruption'}:raise ValueError('Invalid candidate fields')
        identifier(c['id']);artifacts=identifiers(c['artifacts'])
        if c['id'] in seen:raise ValueError('Duplicate candidate')
        seen.add(c['id'])
        if not isinstance(c['covers'],list) or any(v not in ('task','support','review') for v in c['covers']):raise ValueError('Invalid coverage')
        if type(c['disruption']) is not int or not 0<=c['disruption']<=100:raise ValueError('Invalid disruption cost')
        check=ledger.check(artifacts,purpose,recipient,operation)
        eligible=check['decision']=='ALLOW' and required<=set(c['covers'])
        considered.append({'id':c['id'],'eligible':eligible,'permission_decision':check['decision'],
                           'missing_coverage':sorted(required-set(c['covers'])),'disruption':c['disruption']})
    eligible=[c for c in considered if c['eligible']]
    selected=min(eligible,key=lambda c:(c['disruption'],c['id']))['id'] if eligible else None
    return {'status':'PROPOSAL_READY' if selected else 'NO_PERMITTED_RESPONSE','selected':selected,
            'requirements':sorted(required),'candidates':considered,'coverage_source':'host_declaration',
            'semantic_sufficiency_verified':False,'execution_authorized':False,'external_action_taken':False,
            'ledger_revision':ledger.revision,'recheck_required_before_execution':True}


class _SessionModel:
    """Bounded session-only memoization of immutable prefix interpretations."""
    def __init__(self,model):
        self.model=model;self.identity=model.identity;self.cache={}
    def predict(self,texts,language):
        keys=[hashlib.sha256(json.dumps([language,t],sort_keys=True,ensure_ascii=False).encode()).hexdigest() for t in texts]
        missing={key:text for key,text in zip(keys,texts) if key not in self.cache}
        if len(self.cache)+len(missing)>48:raise ValueError('Evidence session cache limit')
        if missing:
            predictions=self.model.predict(list(missing.values()),language)
            if len(predictions)!=len(missing):raise ValueError('Incomplete model output')
            self.cache.update(zip(missing,deepcopy(predictions)))
        return [deepcopy(self.cache[key]) for key in keys]
    def close(self):self.cache.clear()


class EvidenceSession:
    """Ephemeral integrated evidence and permission state, owned by the host.

    Model hypotheses never update source permissions. A paragraph remains tied
    to its complete parent turn when constructing a derived response artifact.
    """
    def __init__(self,language,model=None):
        from .flow import FlowLedger
        if language not in ('ro','uk'):raise ValueError('Unsupported language')
        self.language=language;self.model=_SessionModel(model or EvidenceCascade())
        self.ledger=FlowLedger();self.state=EvidenceState();self._turns=[];self._closed=False

    def append(self,role,text):
        if self._closed:raise ValueError('Closed evidence session')
        proposed=self._turns+[{'role':role,'text':text}]
        report=analyze_evidence({'language':self.language,'turns':proposed},self.model)
        # Validate append-only state before mutating either owned structure.
        trial=deepcopy(self.state);trial.update(report)
        source_id='turn_'+str(len(self._turns));self.ledger.add_source(source_id,text)
        self._turns=proposed;self.state=trial
        return {'source_id':source_id,'state':self.state.snapshot(),'external_action_taken':False}

    def derive_from_evidence(self,artifact_id,text,evidence_ids,*,kind='transformation'):
        """Host-provided response text inherits complete parent-turn restrictions."""
        from .flow import identifiers
        if self._closed:raise ValueError('Closed evidence session')
        evidence_ids=identifiers(evidence_ids,'evidence ids',maximum=48)
        report=self.state.snapshot()['report']
        if report is None:raise ValueError('No evidence yet')
        by_id={e['id']:e for e in report['evidence']}
        if any(i not in by_id for i in evidence_ids):raise ValueError('Unknown evidence reference')
        parents=sorted({'turn_'+str(by_id[i]['reference']['turn_index']) for i in evidence_ids})
        self.ledger.derive(artifact_id,text,parents,kind=kind)
        return self.ledger.describe(artifact_id)

    def propose_response(self,candidates,*,purpose,recipient,operation='use'):
        if self._closed:raise ValueError('Closed evidence session')
        snapshot=self.state.snapshot()
        if snapshot['report'] is None:raise ValueError('No evidence yet')
        report=deepcopy(snapshot['report']);superseded={r['old_id'] for r in snapshot['revisions']}
        report['evidence']=[e for e in report['evidence'] if e['id'] not in superseded]
        report['review_ids']=snapshot['active_review_ids']
        proposal=plan_response(report,candidates,self.ledger,purpose=purpose,recipient=recipient,operation=operation)
        proposal['host_reviewed_revisions_applied']=len(snapshot['revisions'])
        return proposal

    def close(self):
        self.state.close();self.ledger.close();self.model.close();self._turns.clear();self._closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def main():
    p=argparse.ArgumentParser(description='Experimental Romanian/Ukrainian structured evidence proposals')
    p.add_argument('input',type=Path);p.add_argument('--backend',choices=['character','semantic','guarded'],default='character');p.add_argument('--cascade',action='store_true');p.add_argument('--model-path',type=Path);p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    a=p.parse_args()
    from .cli import _strict_json_loads
    try:
        if a.input.stat().st_size>64000:raise ValueError('Input too large')
        payload=_strict_json_loads(a.input.read_text());validate(payload)
        if a.cascade and a.backend!='character':raise ValueError('Cascade uses the character primary; omit --backend semantic')
        challenger=EvidenceModel('semantic',base_path=a.model_path,device=a.device) if a.cascade else None
        primary=load_evidence_model(a.backend,base_path=a.model_path,device=a.device)
        report=analyze_evidence(payload,EvidenceCascade(primary,challenger))
    except (ValueError,OSError,ImportError) as e:p.error(str(e))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
