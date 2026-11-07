"""Portable, explicitly experimental v10 student and evidence selector.

Inference uses the standard library. This lexical student is a measured baseline,
not a replacement for the qualified v9 Gemma model. Optional training lives in
training/train_trajectory_v10.py. No external model calls or implicit downloads.
"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from .assurance import assure, digest, policy_digest
from .evidence import ONTOLOGY
from .trajectory_data import prefix, input_identity
from .trajectory_memory import select, render
from .trajectory_sets import WORLDS, predict_set

DIMENSION = 4096
RESPONSE_LABELS = ('appropriate', 'boundary', 'privacy', 'pressure', 'unsupported')


def features(text, language, role='context'):
    value = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text).casefold())
    tokens = ['lang:'+language, 'role:'+role]
    tokens += ['w:'+word for word in re.findall(r'\w+', value)]
    tokens += ['c:'+value[i:i+n] for n in (3, 4) for i in range(len(value)-n+1)]
    counts = Counter(int.from_bytes(hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest(), 'big') % DIMENSION for token in tokens)
    out = {k: 1+math.log(v) for k, v in counts.items()}
    norm = math.sqrt(sum(v*v for v in out.values()))
    return {k: v/norm for k, v in out.items()} if norm else {}


def head_predict(head, vector):
    logits = [bias+sum(weights[i]*value for i, value in vector.items()) for weights, bias in zip(head['weights'], head['bias'])]
    if len(logits) == 1 and len(head['classes']) == 2:
        z = max(-700, min(700, logits[0]))
        p = 1/(1+math.exp(-z))
        return dict(zip(head['classes'], (1-p, p)))
    top = max(logits)
    exp = [math.exp(v-top) for v in logits]
    total = sum(exp)
    return dict(zip(head['classes'], [v/total for v in exp]))


def view_features(view, mode='pooled'):
    if mode=='pooled':return features(render(view),view['language'])
    if mode!='focus-context':raise ValueError('Unknown view representation')
    focal='\n'.join(s['text'] for s in view['spans'] if s['turn']==view['focus'])
    history='\n'.join('['+s['role']+'] '+s['text'] for s in view['spans'] if s['turn']!=view['focus'])
    values={}
    for offset,text in ((0,focal),(DIMENSION//2,history)):
        for i,v in features(text,view['language']).items():
            index=i%(DIMENSION//2)+offset
            values[index]=values.get(index,0)+v
    norm=math.sqrt(sum(v*v for v in values.values()))
    return {i:v/norm for i,v in values.items()}


def world_probabilities(heads):
    probabilities=[math.prod(max(heads[k].get(w[k],0),1e-12) for k in ONTOLOGY) for w in WORLDS]
    total=sum(probabilities);probabilities=[v/total for v in probabilities]
    if 'joint' in heads:
        probabilities=[.95*heads['joint'].get(str(i),0)+.05*p for i,p in enumerate(probabilities)]
    return probabilities


def review_score(heads, probabilities, mode='binary'):
    if mode=='binary':return heads['review'].get('1',0)
    if mode!='joint-policy':raise ValueError('Unknown review scoring mode')
    from .evidence import _complete_action
    return sum(p for w,p in zip(WORLDS,probabilities) if _complete_action(w)!='CONTINUE')


class TrajectoryModel:
    def __init__(self, path, *, expected_sha256=None):
        raw = Path(path).read_bytes()
        self.identity = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and expected_sha256 != self.identity:
            raise ValueError('Model artifact digest differs')
        if len(raw) > 20_000_000:
            raise ValueError('Model artifact exceeds size limit')
        with gzip.open(path, 'rt', encoding='utf-8') as source:
            decoded = source.read(40_000_001)
        if len(decoded) > 40_000_000:
            raise ValueError('Decompressed model exceeds limit')
        self.spec = json.loads(decoded)
        if self.spec.get('format') != 'roguard-student-v10-research1' or self.spec.get('dimension') != DIMENSION:
            raise ValueError('Unsupported student artifact')
        expected = {**{k: set(v) for k, v in ONTOLOGY.items()}, 'review': {'0', '1'}, 'relevance': {'0', '1'}, 'response': set(RESPONSE_LABELS)}
        if 'joint' in self.spec['heads']:
            expected['joint']={str(i) for i in range(len(WORLDS))}
        if set(self.spec['heads']) != set(expected):
            raise ValueError('Missing prediction heads')
        for name, head in self.spec['heads'].items():
            classes = head['classes']
            if not classes or len(set(classes)) != len(classes) or not set(classes) <= expected[name]:
                raise ValueError('Invalid head classes')
            size = len(classes)
            if len(head['weights']) not in ({1, 2} if size == 2 else {size}) or len(head['bias']) != len(head['weights']):
                raise ValueError('Invalid classifier shape')
            for weights in head['weights']:
                if len(weights) != DIMENSION or any(type(v) not in (int, float) or not math.isfinite(v) for v in weights):
                    raise ValueError('Invalid classifier weights')
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in head['bias']):
                raise ValueError('Invalid classifier intercept')
        self.threshold = self.spec['review_threshold']
        if type(self.threshold) not in (int, float) or not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid review threshold')

    def relevance(self, text, language, role):
        return head_predict(self.spec['heads']['relevance'], features(text, language, role)).get('1', 0)

    def pipeline_digest(self, budget=3000, strategy='learned'):
        contract={'model': self.identity, 'budget': budget, 'strategy': strategy,
                       'feature_contract': 'blake2b-word-char34-l2-v1', 'policy': policy_digest(),
                       'memory_contract': 'sentence-offsets-focal-mandatory-v1'}
        if 'min_relevance' in self.spec:
            contract['min_relevance']=self.spec['min_relevance']
        if 'representation' in self.spec:
            contract['representation']=self.spec['representation']
        if 'review_mode' in self.spec:
            contract['review_mode']=self.spec['review_mode']
        return digest(contract)

    def predict(self, payload, *, focus=None, budget=3000, strategy='learned', calibration=None, plan=False):
        focus = len(payload['turns'])-1 if focus is None else focus
        payload = prefix(payload, focus)
        view = select(payload, self.relevance, budget=budget, strategy=strategy,min_relevance=self.spec.get('min_relevance'))
        vector = view_features(view,self.spec.get('representation','pooled'))
        heads = {k: head_predict(v, vector) for k, v in self.spec['heads'].items() if k not in ('relevance', 'response')}
        # Unseen ontology values retain a small probability; a conservative
        # quantile of one therefore expands to the entire finite ontology.
        probabilities=world_probabilities(heads)
        score=review_score(heads,probabilities,self.spec.get('review_mode','binary'))
        binary = None if view['mandatory_overflow'] else int(score >= self.threshold)
        pipeline = self.pipeline_digest(budget, strategy)
        result = {'format': 'roguard-trajectory-prediction-v10-research1',
                  'language': payload['language'], 'focus': focus, 'input_digest': input_identity(payload),
                  'model_identity': self.identity, 'pipeline_digest': pipeline,
                  'predicted': binary, 'score': score, 'probabilities': probabilities,
                  'evidence': [{k: v for k, v in s.items() if k not in ('text', 'role')} for s in view['spans']],
                  'memory': {k: v for k, v in view.items() if k != 'spans'},
                  'qualified': False, 'establishes_safety': False, 'permits_external_action': False}
        if calibration is not None:
            sets = predict_set(probabilities, calibration, language=payload['language'], pipeline_digest=pipeline,
                               supported=not view['mandatory_overflow'])
            result['interpretation_set'] = sets
            if plan:
                result['assurance'] = assure({'language': payload['language'], 'context_digest': result['input_digest'],
                    'hypotheses': sets['hypotheses'], 'model_digests': [self.identity], 'review_floor': binary == 1})
        return result

    def review_response(self, draft, language, context=''):
        if language not in ('ro', 'uk') or not isinstance(draft, str) or not 1 <= len(draft.strip()) <= 6000 or not isinstance(context, str) or len(context) > 12000:
            raise ValueError('Invalid response input')
        values = head_predict(self.spec['heads']['response'], features(context+'\n[DRAFT] '+draft, language))
        return {'label': max(values, key=values.get), 'scores': values, 'model_identity': self.identity,
                'qualified': False, 'safe_response_established': False, 'permits_external_action': False}


class TrajectorySession:
    """An explicit host-owned review floor; topic changes never close a concern."""
    def __init__(self, model):
        self.model = model
        self._last = None
        self._open = set()

    def predict(self, payload, **kwargs):
        if self._last is not None and (payload['language'] != self._last['language'] or payload['turns'][:len(self._last['turns'])] != self._last['turns'] or len(payload['turns']) < len(self._last['turns'])):
            raise ValueError('Session history changed; start a new session')
        result = self.model.predict(payload, **kwargs)
        if kwargs.get('focus') is not None and kwargs['focus'] != len(payload['turns'])-1:
            raise ValueError('Session predictions must address the latest turn')
        self._last = json.loads(json.dumps(payload))
        if result['predicted'] == 1:
            self._open.add(result['input_digest'])
        result['unresolved_review_references'] = sorted(self._open)
        result['session_review_required'] = bool(self._open) or result['predicted'] is None
        return result

    def resolve(self, reference, *, expected_context_digest):
        """Trusted host/reviewer operation, never a model-exposed tool."""
        if self._last is None or input_identity(self._last) != expected_context_digest or reference not in self._open:
            raise ValueError('Stale or unknown reviewer resolution')
        self._open.remove(reference)
