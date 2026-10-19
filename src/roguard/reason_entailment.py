"""Support/revision proposals with separate development-selected cutoffs."""
import json
import math
from pathlib import Path
import threading
from .paired_nli import prepare,PREPROCESSING
from .native_review import _read,_sha,_digest
from .native_assets import reject_native_overrides
from .nli_v21 import MODEL_ID,REVISION,BASE_HASHES

from .policy_entailment import _PolicyEngine,HYPOTHESES


class ReasonEntailmentReview:
    def __init__(self,artifact,checkpoint,base,*,expected_sha256,device='cpu'):
        if device not in ('cpu','mps','cuda'):raise ValueError('Unsupported model device')
        raw=_read(artifact,65536)
        if _sha(raw)!=_digest(expected_sha256):raise ValueError('Policy-entailment artifact hash differs')
        spec=json.loads(raw);required={'format','base','revision','thresholds','hypotheses','checkpoint_sha256','base_hashes','score','preprocessing'}
        if not isinstance(spec,dict) or set(spec)!=required or spec['format']!='roguard-reason-entailment-artifact-v1':raise ValueError('Invalid policy-entailment artifact')
        if (spec['base'],spec['revision'],spec['hypotheses'],spec['score'],spec['preprocessing'])!=(MODEL_ID,REVISION,list(HYPOTHESES),'separate-entailment-thresholds-including-neutral',PREPROCESSING):raise ValueError('Policy-entailment contract differs')
        if spec['base_hashes']!=BASE_HASHES:raise ValueError('Base inventory differs')
        limits=spec['thresholds']
        if not isinstance(limits,dict) or set(limits)!={'ro','uk'}:raise ValueError('Both language thresholds required')
        for values in limits.values():
            if not isinstance(values,dict) or set(values)!={'support','revision'}:raise ValueError('Both reason thresholds required')
            if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values.values()):raise ValueError('Invalid threshold')
        checkpoint_sha=_digest(spec['checkpoint_sha256']);reject_native_overrides(base,BASE_HASHES);files={}
        for name,expected in BASE_HASHES.items():
            if name=='model.safetensors':continue
            data=_read(Path(base)/name,40_000_000)
            if _sha(data)!=expected:raise ValueError('Pinned configuration/tokenizer changed: '+name)
            files[name]=data
        self._identity=expected_sha256;self._thresholds={lang:dict(values) for lang,values in limits.items()};self._args=(files,Path(checkpoint),checkpoint_sha,device);self._engine=None;self._lock=threading.RLock()

    @property
    def identity(self):return self._identity

    @property
    def thresholds(self):return {lang:dict(values) for lang,values in self._thresholds.items()}

    def predict(self,payload,*,focus=None):return self.predict_many([payload],focuses=[focus])[0]

    def predict_many(self,payloads,*,focuses=None):
        if not isinstance(payloads,list) or not 1<=len(payloads)<=128:raise ValueError('Expected one to128 conversations')
        if focuses is None:focuses=[None]*len(payloads)
        if not isinstance(focuses,list) or len(focuses)!=len(payloads):raise ValueError('Focal indices differ')
        prepared=[prepare(p,f) for p,f in zip(payloads,focuses)]
        with self._lock:
            if self._engine is None:self._engine=_PolicyEngine(*self._args)
            scores=self._engine.scores([p[0] for p in prepared])
        if not isinstance(scores,list) or len(scores)!=len(prepared):raise ValueError('Incomplete policy-entailment output')
        results=[]
        for values,(_,language,focus,identity) in zip(scores,prepared):
            if values is not None and (not isinstance(values,(list,tuple)) or len(values)!=2 or any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in values)):raise ValueError('Invalid policy-entailment scores')
            score=None if values is None else max(values);limit=self._thresholds[language]
            support=None if values is None else values[0]>=limit['support'];revision=None if values is None else values[1]>=limit['revision']
            predicted=None if values is None else int(support or revision)
            reason=('MODEL_CONTEXT_LIMIT' if values is None else 'SUPPORT_AND_REVISION_PROPOSAL' if support and revision else 'PERSONAL_SUPPORT_PROPOSAL' if support else 'REVISION_REVIEW_PROPOSAL' if revision else 'NO_SIGNAL_DETECTED')
            results.append(dict(format='roguard-reason-entailment-review-v1',language=language,focus=focus,input_digest=identity,model_identity=self.identity,predicted=predicted,score=score,support_score=None if values is None else values[0],revision_score=None if values is None else values[1],support_review_requested=support,revision_review_requested=revision,decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',reason=reason,review_requested=predicted!=0,score_is_calibrated=False,score_kind='maximum_conditioned_entailment_probability',decision_rule='support_or_revision_at_separate_thresholds',support_threshold=limit['support'],revision_threshold=limit['revision'],text_in_report=False,establishes_safety=False,permits_external_action=False,permits_notification=False))
        return results
