"""Distinct support/revision proposals under the existing human-review policy."""
import json
import math
from pathlib import Path
import threading
from .paired_nli import _Engine,prepare,PREPROCESSING
from .native_review import _read,_sha,_digest
from .native_assets import reject_native_overrides
from .nli_v21 import MODEL_ID,REVISION,BASE_HASHES

HYPOTHESES=(
 'The focal message conveys a real person\'s emotional distress, safety problem or need for personal support, including uncertain, past or hypothetical concerns.',
 'The focal message withdraws an earlier concerning statement, or gives conflicting claims about whether that concerning statement is real or fictional.',
)


class _PolicyEngine(_Engine):
    def scores(self,premises):
        encoded=self.tokenizer([p for p in premises for _ in HYPOTHESES],list(HYPOTHESES)*len(premises),padding=False,truncation=False)
        valid=[i for i in range(len(premises)) if all(len(encoded['input_ids'][2*i+t])<=512 for t in range(2))]
        values=[None]*len(premises)
        with self.torch.inference_mode():
            for start in range(0,len(valid),4):
                group=valid[start:start+4];indices=[2*i+t for i in group for t in range(2)]
                batch=self.tokenizer.pad([{k:v[i] for k,v in encoded.items()} for i in indices],return_tensors='pt',pad_to_multiple_of=8)
                logits=self.model(**{k:v.to(self.device) for k,v in batch.items()}).logits.float()
                if tuple(logits.shape)!=(len(indices),3):raise ValueError('Invalid policy-entailment output shape')
                scores=self.torch.softmax(logits,dim=-1)[:,0].cpu().tolist()
                for j,i in enumerate(group):values[i]=scores[2*j:2*j+2]
        return values


class PolicyEntailmentReview:
    def __init__(self,artifact,checkpoint,base,*,expected_sha256,device='cpu'):
        if device not in ('cpu','mps','cuda'):raise ValueError('Unsupported model device')
        raw=_read(artifact,65536)
        if _sha(raw)!=_digest(expected_sha256):raise ValueError('Policy-entailment artifact hash differs')
        spec=json.loads(raw);required={'format','base','revision','thresholds','hypotheses','checkpoint_sha256','base_hashes','score','preprocessing'}
        if not isinstance(spec,dict) or set(spec)!=required or spec['format']!='roguard-policy-entailment-artifact-v1':raise ValueError('Invalid policy-entailment artifact')
        if (spec['base'],spec['revision'],spec['hypotheses'],spec['score'],spec['preprocessing'])!=(MODEL_ID,REVISION,list(HYPOTHESES),'maximum-of-two-entailment-probabilities-including-neutral',PREPROCESSING):raise ValueError('Policy-entailment contract differs')
        if spec['base_hashes']!=BASE_HASHES:raise ValueError('Base inventory differs')
        limits=spec['thresholds']
        if not isinstance(limits,dict) or set(limits)!={'ro','uk'}:raise ValueError('Both language thresholds required')
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in limits.values()):raise ValueError('Invalid threshold')
        checkpoint_sha=_digest(spec['checkpoint_sha256']);reject_native_overrides(base,BASE_HASHES);files={}
        for name,expected in BASE_HASHES.items():
            if name=='model.safetensors':continue
            data=_read(Path(base)/name,40_000_000)
            if _sha(data)!=expected:raise ValueError('Pinned configuration/tokenizer changed: '+name)
            files[name]=data
        self._identity=expected_sha256;self._thresholds=dict(limits);self._args=(files,Path(checkpoint),checkpoint_sha,device);self._engine=None;self._lock=threading.RLock()

    @property
    def identity(self):return self._identity

    @property
    def thresholds(self):return dict(self._thresholds)

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
            support=None if values is None else values[0]>=limit;revision=None if values is None else values[1]>=limit
            predicted=None if values is None else int(support or revision)
            reason=('MODEL_CONTEXT_LIMIT' if values is None else 'SUPPORT_AND_REVISION_PROPOSAL' if support and revision else 'PERSONAL_SUPPORT_PROPOSAL' if support else 'REVISION_REVIEW_PROPOSAL' if revision else 'NO_SIGNAL_DETECTED')
            results.append(dict(format='roguard-policy-entailment-review-v1',language=language,focus=focus,input_digest=identity,model_identity=self.identity,predicted=predicted,score=score,support_score=None if values is None else values[0],revision_score=None if values is None else values[1],support_review_requested=support,revision_review_requested=revision,decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',reason=reason,review_requested=predicted!=0,score_is_calibrated=False,score_kind='maximum_conditioned_entailment_probability',text_in_report=False,establishes_safety=False,permits_external_action=False,permits_notification=False))
        return results
