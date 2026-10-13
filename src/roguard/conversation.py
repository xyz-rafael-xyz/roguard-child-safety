"""Local conversation advisory and independently declared action checks."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import sys
from functools import lru_cache
from importlib import resources
from pathlib import Path
from .cli import _strict_json_loads, assess_json
from .message_model import SyntheticMessageModel
from .response_review import review_response
from .v2_audit import audit_v2_contract

MAX_TURNS=16
MAX_TOTAL_CHARS=12000
ROLES={'minor','assistant','adult','other','unknown'}
STATUSES={'REVIEW_SUPPORT_SIGNAL','NO_MODEL_SIGNAL','INSUFFICIENT_EVIDENCE'}

def render(turns):
    return '\n'.join(f"{turn['role']}: {turn['text']}" for turn in turns)

@lru_cache(maxsize=2)
def fast_model(language):
    return SyntheticMessageModel(resources.read_binary('roguard',f'conversation-model-{language}.json.gz'))

@lru_cache(maxsize=1)
def configuration():
    return json.loads(resources.read_text('roguard','conversation-selection.json'))

class ConversationScorer:
    """Owns cached weights, never a persistent cache of messages or embeddings."""
    def __init__(self, backend='fast', base_path=None, device='cpu'):
        if backend not in ('fast','semantic','hybrid','adapted'):
            raise ValueError('Unknown backend')
        self.backend=backend
        self.encoder=None
        if backend!='fast':
            if base_path is None:
                from .model_setup import locate_model
                base_path=locate_model()
            from .semantic import SemanticEncoder
            self.encoder=SemanticEncoder(base_path,device)
        self.config=configuration()
        self.head=json.loads(resources.read_text('roguard','conversation-head.json')) if self.encoder else None
        self.head_sha256=hashlib.sha256(resources.read_binary('roguard','conversation-head.json')).hexdigest() if self.encoder else None
        if backend=='adapted':
            from .adapted import load_adapted,ADAPTER_SHA256
            self.head,self.model_id,self.calibrators=load_adapted(self.encoder)
            self.head_sha256=ADAPTER_SHA256

    def scores(self, texts, language):
        # The legacy message API has a 2000-character limit; the conversation
        # feature scorer has an explicitly separate bounded input contract.
        model=fast_model(language)
        lexical=[_long_score(model,text) for text in texts]
        if self.encoder is None:return lexical
        from .semantic import head_scores
        semantic=head_scores(self.encoder.encode(texts),self.head).tolist()
        weight=1.0 if self.backend in ('semantic','adapted') else self.config['semantic_weight']
        return [weight*s+(1-weight)*c for s,c in zip(semantic,lexical)]

    def thresholds(self, language):
        if self.backend=='adapted':return {'cutoff':self.head['cutoffs'][language],'abstain_margin':.08}
        section='fast' if self.backend=='fast' else self.backend
        return self.config[section][language]


def _long_score(model, text):
    # Keep the original v3 public input limit unchanged; reuse its exact math.
    from collections import Counter
    import unicodedata
    normalized=re.sub(r'\s\s+', ' ', unicodedata.normalize('NFC',text).lower())
    counts=Counter(normalized[i:i+n] for n in range(model.low,model.high+1)
                   for i in range(len(normalized)-n+1))
    weighted=[((1+math.log(count))*model.features[name][0],model.features[name][1])
              for name,count in counts.items() if name in model.features]
    norm=math.sqrt(sum(value*value for value,_ in weighted))
    logit=model.intercept+(sum(value*w for value,w in weighted)/norm if norm else 0)
    return 1/(1+math.exp(-max(-700,min(700,logit))))

@lru_cache(maxsize=1)
def default_scorer():return ConversationScorer()


def validate(payload):
    if not isinstance(payload,dict) or set(payload)-{'language','turns','draft','revision','contract','audit','case_id'}:
        raise ValueError('Unknown conversation field')
    if not isinstance(payload.get('language'),str) or len(payload['language'])>16:
        raise ValueError('Declare the language')
    turns=payload.get('turns')
    if not isinstance(turns,list) or not 1<=len(turns)<=MAX_TURNS:
        raise ValueError('Provide 1 to 16 turns')
    for turn in turns:
        if not isinstance(turn,dict) or set(turn)!={'role','text'} or not isinstance(turn.get('role'),str) or turn['role'] not in ROLES:
            raise ValueError('Each turn needs a supported declared role and text')
        if not isinstance(turn['text'],str) or (not turn['text'].strip() or len(turn['text'])>2000):
            raise ValueError('Each turn must have 1 to 2000 characters')
    if sum(len(t['text']) for t in turns)>MAX_TOTAL_CHARS:raise ValueError('Conversation too long')
    if 'case_id' in payload and (not isinstance(payload['case_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',payload['case_id'])):
        raise ValueError('Case ID must be a short opaque identifier')
    for field in ('draft','revision'):
        if field in payload and (not isinstance(payload[field],str) or (not payload[field].strip() or len(payload[field])>6000)):
            raise ValueError('Drafts must contain 1 to 6000 characters')
    if 'revision' in payload and 'draft' not in payload:raise ValueError('Revision requires an original draft')
    for field in ('contract','audit'):
        if field in payload and (not isinstance(payload[field],dict) or payload[field].get('language')!=payload['language']):
            raise ValueError('Declared contract language must match')
    return payload


def language_uncertainty(turns, language):
    if language not in ('ro','uk'):return ['UNSUPPORTED_LANGUAGE']
    text=' '.join(t['text'] for t in turns if t['role']=='minor')
    letters=[c for c in text if c.isalpha()]
    if len(letters)<10:return ['TOO_LITTLE_LANGUAGE_EVIDENCE']
    cyr=sum('\u0400'<=c<='\u052f' for c in letters)/len(letters)
    if (language=='ro' and cyr>.2) or (language=='uk' and cyr<.6):
        return ['SCRIPT_MISMATCH_OR_MIXTURE']
    words=set(re.findall(r'[^\W\d_]+',text.lower()))
    if language=='ro' and not words.intersection({'eu','nu','am','sunt','să','sa','și','si','de','la','cu','mă','ma','îmi','imi','vreau','ce','un','o','este','e','mi','aș','as'}):
        return ['LANGUAGE_NOT_CONFIRMED']
    if len(words.intersection({'please','need','help','with','feel','someone','would','about'}))>=2:
        return ['LANGUAGE_MIXTURE_REQUIRES_REVIEW']
    return []


def analyze_conversation(payload, scorer=None):
    payload=validate(payload);language=payload['language'];turns=payload['turns']
    reasons=language_uncertainty(turns,language)
    if not any(t['role']=='minor' for t in turns):reasons.append('NO_DECLARED_MINOR_TURN')
    if any(t['role'] in ('adult','other','unknown') for t in turns):reasons.append('UNSUPPORTED_ROLE_CONTEXT')
    decision='INSUFFICIENT_EVIDENCE';score=None;cutoff=None;evidence=[];identity=None
    if not reasons:
        scorer=scorer or default_scorer()
        settings=scorer.thresholds(language);cutoff=settings['cutoff'];margin=settings['abstain_margin']
        try:
            score=float(scorer.scores([render(turns)],language)[0])
            if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid model score')
            decision=('INSUFFICIENT_EVIDENCE' if abs(score-cutoff)<=margin else
                      'REVIEW_SUPPORT_SIGNAL' if score>=cutoff else 'NO_MODEL_SIGNAL')
            if decision=='INSUFFICIENT_EVIDENCE':reasons.append('NEAR_DEVELOPMENT_THRESHOLD')
            if getattr(scorer,'calibrators',None):
                calibrated=scorer.calibrators[language].predict(score,model_id=scorer.model_id,language=language)
                decision=calibrated['decision'];reasons=['CALIBRATED_SET_NOT_SINGLETON'] if len(calibrated['prediction_set'])!=1 else []
            # Turn ablation is an inspectable sensitivity probe, not a causal explanation.
            indices=[i for i,t in enumerate(turns) if t['role']=='minor']
            variants=[render([{**t,'text':''} if i==removed else t for i,t in enumerate(turns)]) for removed in indices]
            ablated=scorer.scores(variants,language)
            for i,value in zip(indices,ablated):
                evidence.append({'turn_index':i,'start':0,'end':len(turns[i]['text']),
                                 'score_change_if_omitted':round(score-float(value),6),
                                 'basis':'turn_ablation_sensitivity_not_verified_evidence'})
            evidence.sort(key=lambda e:-abs(e['score_change_if_omitted']))
            evidence=evidence[:3]
            identity={'backend':scorer.backend,'semantic_preprocessing':'surface-v1' if scorer.backend!='fast' else None,'fast_sha256':fast_model(language).sha256,
                      'semantic_head_sha256':getattr(scorer,'head_sha256',None),
                      'base_revision':configuration().get('base_revision') if scorer.backend!='fast' else None}
        except ValueError:
            decision='INSUFFICIENT_EVIDENCE';score=None;evidence=[];reasons.append('MODEL_INPUT_UNSUPPORTED')
    report={'schema_version':'4.0','language':language,'case_id':payload.get('case_id'),
        'signal':{'decision':decision,'reason_codes':reasons,'synthetic_score':round(score,6) if score is not None else None,
                  'development_cutoff':cutoff,'model':identity,'evidence_references':evidence},
        'response_review':review_response(payload['draft'],language,payload.get('revision')) if 'draft' in payload and language in ('ro','uk') else None,
        'declared_contract':assess_json(payload['contract']) if 'contract' in payload else None,
        'uncertainty_audit':audit_v2_contract(payload['audit']) if 'audit' in payload else None,
        'next_step':'HUMAN_REVIEW' if decision!='NO_MODEL_SIGNAL' else 'NO_MODEL_SIGNAL_DOES_NOT_CLOSE_CONCERN',
        'authority_from_model':False,'external_action_taken':False,'text_in_report':False,
        'real_world_accuracy_established':False,'language_verified':False}
    if 'calibrated' in locals():report['calibration']=calibrated
    report['action_review']=action_review(report['declared_contract'],report['uncertainty_audit'])
    report['trace']=[{'step':'interpretation','source':'model_advisory','decision':decision},
                     {'step':'proposed_action','source':'declared_facts_only',
                      'decision':report['action_review']['status']}]
    report['declared_facts_unverified']=True
    return report


def action_review(contract,audit):
    reasons=[]
    if contract is None and audit is None:
        return {'status':'NOT_ASSESSED','reason_codes':['NO_ACTION_FACTS'],'external_action_taken':False}
    if contract is None or not contract.get('proposed_use'):
        reasons.append('MISSING_COMBINED_ACTION_CONTRACT')
    elif contract['proposed_use']['status']!='ready_for_human_decision':
        reasons.append('DECLARED_ACTION_REQUIREMENTS_NOT_MET')
    if contract and any(f['status']!='pass' for f in contract['findings']):
        reasons.append('DECLARED_CHECK_REQUIRES_REVIEW')
    routing=next((c for c in (audit or {}).get('categories',[]) if c['category']=='R1'),None)
    if routing is None:
        reasons.append('MISSING_RECIPIENT_CONCERN_CHECK')
    elif routing['decision_stability']!='stable' or routing['possible_decision_codes']!=['NO_ROUTE_SIGNAL']:
        reasons.append('RECIPIENT_CONCERN_OR_SCOPE_UNRESOLVED')
    return {'status':'HUMAN_REVIEW_REQUIRED' if reasons else 'READY_FOR_HUMAN_REVIEW',
            'reason_codes':reasons,'authority_verified':False,'external_action_taken':False}


def analyze_batch(payloads, scorer=None):
    if not isinstance(payloads,list) or not 1<=len(payloads)<=200:raise ValueError('Batch must contain 1 to 200 conversations')
    for payload in payloads:validate(payload)
    scorer=scorer or default_scorer()
    # Request-scoped reuse only; text keys and embeddings are discarded on return.
    cached={}
    for language in ('ro','uk'):
        texts=[]
        for payload in payloads:
            turns=payload['turns']
            if payload['language']!=language or language_uncertainty(turns,language) or any(t['role'] not in ('minor','assistant') for t in turns):
                continue
            texts.append(render(turns))
            texts.extend(render([{**t,'text':''} if i==removed else t for i,t in enumerate(turns)])
                         for removed,t in enumerate(turns) if t['role']=='minor')
        texts=list(dict.fromkeys(texts))
        if texts:
            try:
                cached.update({(language,text):score for text,score in zip(texts,scorer.scores(texts,language))})
            except ValueError:
                pass  # Individual token-limit errors become individual abstentions.
    class RequestScorer:
        backend=scorer.backend
        head_sha256=getattr(scorer,'head_sha256',None)
        calibrators=getattr(scorer,'calibrators',None)
        model_id=getattr(scorer,'model_id',None)
        def thresholds(self,language):return scorer.thresholds(language)
        def scores(self,texts,language):
            if all((language,text) in cached for text in texts):
                return [cached[(language,text)] for text in texts]
            return scorer.scores(texts,language)
    return [analyze_conversation(p,RequestScorer()) for p in payloads]


def compare_reports(before,after):
    def keyed(rows):
        if not isinstance(rows,list) or not 1<=len(rows)<=200:raise ValueError('Invalid report batch')
        mapping={}
        for row in rows:
            if not isinstance(row,dict) or row.get('schema_version')!='4.0':raise ValueError('Expected v4 reports')
            key=row.get('case_id'); signal=row.get('signal')
            if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',key) or key in mapping:
                raise ValueError('Comparison requires unique case IDs')
            if not isinstance(signal,dict) or signal.get('decision') not in STATUSES:raise ValueError('Invalid decision')
            mapping[key]=row
        return mapping
    left,right=keyed(before),keyed(after)
    if left.keys()!=right.keys():raise ValueError('Comparison case IDs must match')
    changes=[]
    for key in sorted(left):
        if left[key]['language']!=right[key]['language']:raise ValueError('Comparison languages must match')
        a,b=left[key]['signal']['decision'],right[key]['signal']['decision']
        if a!=b:changes.append({'case_id':key,'before':a,'after':b})
    return {'cases':len(left),'decision_changes':changes,'accuracy_change_established':False,
            'text_in_report':False}


def main():
    p=argparse.ArgumentParser(description='Review a local conversation; never sends or stores its text')
    p.add_argument('input',nargs='?',type=Path);p.add_argument('--jsonl',action='store_true')
    p.add_argument('--backend',choices=['fast','semantic','hybrid','adapted'],default='fast')
    p.add_argument('--model-path',type=Path);p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    args=p.parse_args()
    try:
        if args.input:
            with args.input.open(encoding='utf-8') as stream:
                raw=stream.read(MAX_TOTAL_CHARS*1000+1)
        else:
            raw=sys.stdin.read(MAX_TOTAL_CHARS*1000+1)
        if len(raw)>MAX_TOTAL_CHARS*1000:raise ValueError('Input too large')
        scorer=ConversationScorer(args.backend,args.model_path,args.device)
        if args.jsonl:
            reports=analyze_batch([_strict_json_loads(line) for line in raw.splitlines() if line.strip()],scorer)
            print('\n'.join(json.dumps(r,ensure_ascii=False) for r in reports))
        else:print(json.dumps(analyze_conversation(_strict_json_loads(raw),scorer),ensure_ascii=False,indent=2))
    except (ValueError,TypeError,KeyError,OSError,ImportError):p.error('Invalid input or unavailable optional model; see the conversation guide')

if __name__=='__main__':main()


def compare_main():
    parser=argparse.ArgumentParser(description='Compare v4 reports by opaque case ID; no accuracy claim without labels')
    parser.add_argument('before',type=Path);parser.add_argument('after',type=Path)
    args=parser.parse_args()
    def read(path):
        if path.stat().st_size>12_000_000:raise ValueError('Report file too large')
        raw=path.read_text(encoding='utf-8')
        try:
            value=_strict_json_loads(raw)
            return value if isinstance(value,list) else [value]
        except ValueError:return [_strict_json_loads(line) for line in raw.splitlines() if line.strip()]
    try:print(json.dumps(compare_reports(read(args.before),read(args.after)),ensure_ascii=False,indent=2))
    except (ValueError,TypeError,KeyError,OSError):parser.error('Invalid or mismatched report files')
