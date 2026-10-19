"""Offline, hash-bound bilingual entailment review; no action authority."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import threading
from .native_review import prepare_input,_read,_sha,_digest
from .native_assets import reject_native_overrides
from .nli_v21 import MODEL_ID,REVISION,BASE_HASHES
from .semantic import normalize_semantic_text

HYPOTHESIS='In the focal message, a real person expresses distress, a need for emotional support or a safety concern, including a withdrawal of an earlier disclosure.'
PREPROCESSING='normalized-prefix-and-focus-json-v1'


def prepare(payload,focus=None):
    # Validate before copying, and never traverse unselected future turns.
    _,focus,_=prepare_input(payload,focus)
    selected=deepcopy(dict(language=payload['language'],turns=payload['turns'][:focus+1]))
    _,focus,identity=prepare_input(selected,focus)
    turns=[dict(t,text=normalize_semantic_text(t['text'])) for t in selected['turns']]
    premise=json.dumps(dict(previous_messages=turns[:focus],focal_message=turns[focus]),ensure_ascii=False,separators=(',',':'))
    return premise,selected['language'],focus,identity


class _Engine:
    def __init__(self,files,checkpoint,checkpoint_sha,device):
        import torch
        from safetensors.torch import load
        from transformers import AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
        self.torch=torch;self.device=device
        with tempfile.TemporaryDirectory(prefix='roguard-entailment-') as temporary:
            directory=Path(temporary)
            for name,data in files.items():(directory/name).write_bytes(data)
            config=AutoConfig.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
            if config.model_type!='deberta-v2' or config.id2label!={0:'entailment',1:'neutral',2:'contradiction'}:raise ValueError('Entailment architecture differs')
            self.tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
            if self.tokenizer.padding_side!='right':raise ValueError('Right padding required')
            self.model=AutoModelForSequenceClassification.from_config(config,trust_remote_code=False).float()
        raw=_read(checkpoint,1_400_000_000)
        if _sha(raw)!=checkpoint_sha:raise ValueError('Entailment checkpoint hash differs')
        tensors=load(raw);del raw
        for tensor in tensors.values():
            if tensor.is_floating_point() and (tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all()):raise ValueError('Expected finite float32 weights')
        self.model.load_state_dict(tensors,strict=True);del tensors
        self.model=self.model.to(device).eval()

    def scores(self,premises):
        encoded=self.tokenizer(premises,[HYPOTHESIS]*len(premises),padding=False,truncation=False)
        valid=[i for i,ids in enumerate(encoded['input_ids']) if len(ids)<=512];values=[None]*len(premises)
        with self.torch.inference_mode():
            for start in range(0,len(valid),8):
                indices=valid[start:start+8]
                batch=self.tokenizer.pad([{k:v[i] for k,v in encoded.items()} for i in indices],return_tensors='pt',pad_to_multiple_of=8)
                logits=self.model(**{k:v.to(self.device) for k,v in batch.items()}).logits.float()
                if tuple(logits.shape)!=(len(indices),3):raise ValueError('Invalid entailment output shape')
                scores=self.torch.softmax(logits,dim=-1)[:,0].cpu().tolist()
                for i,score in zip(indices,scores):values[i]=score
        return values


class PairedNLIReview:
    def __init__(self,artifact,checkpoint,base,*,expected_sha256,device='cpu'):
        if device not in ('cpu','mps','cuda'):raise ValueError('Unsupported model device')
        raw=_read(artifact,65536)
        if _sha(raw)!=_digest(expected_sha256):raise ValueError('Entailment artifact hash differs')
        spec=json.loads(raw)
        required={'format','base','revision','thresholds','hypothesis','checkpoint_sha256','base_hashes','score','preprocessing'}
        if not isinstance(spec,dict) or set(spec)!=required or spec['format']!='roguard-paired-nli-artifact-v1':raise ValueError('Invalid entailment artifact')
        if (spec['base'],spec['revision'],spec['hypothesis'],spec['score'],spec['preprocessing'])!=(MODEL_ID,REVISION,HYPOTHESIS,'softmax-entailment-including-neutral',PREPROCESSING):raise ValueError('Entailment contract differs')
        if spec['base_hashes']!=BASE_HASHES:raise ValueError('Entailment base inventory differs')
        limits=spec['thresholds']
        if not isinstance(limits,dict) or set(limits)!={'ro','uk'}:raise ValueError('Both language thresholds required')
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in limits.values()):raise ValueError('Invalid entailment threshold')
        checkpoint_sha=_digest(spec['checkpoint_sha256']);reject_native_overrides(base,BASE_HASHES)
        files={}
        for name,expected in BASE_HASHES.items():
            if name=='model.safetensors':continue
            data=_read(Path(base)/name,40_000_000)
            if _sha(data)!=expected:raise ValueError('Pinned entailment configuration/tokenizer changed: '+name)
            files[name]=data
        self.identity=expected_sha256;self.thresholds=dict(limits);self._args=(files,Path(checkpoint),checkpoint_sha,device);self._engine=None;self._lock=threading.RLock()

    def predict(self,payload,*,focus=None):return self.predict_many([payload],focuses=[focus])[0]

    def predict_many(self,payloads,*,focuses=None):
        if not isinstance(payloads,list) or not 1<=len(payloads)<=128:raise ValueError('Expected one to128 conversations')
        if focuses is None:focuses=[None]*len(payloads)
        if not isinstance(focuses,list) or len(focuses)!=len(payloads):raise ValueError('Focal indices differ')
        prepared=[prepare(payload,focus) for payload,focus in zip(payloads,focuses)]
        with self._lock:
            if self._engine is None:self._engine=_Engine(*self._args)
            scores=self._engine.scores([p[0] for p in prepared])
        if not isinstance(scores,list) or len(scores)!=len(prepared):raise ValueError('Incomplete entailment outputs')
        results=[]
        for score,(_,language,focus,identity) in zip(scores,prepared):
            if score is not None and (type(score) not in (float,int) or not math.isfinite(score) or not 0<=score<=1):raise ValueError('Invalid entailment score')
            predicted=None if score is None else int(score>=self.thresholds[language])
            results.append(dict(format='roguard-paired-nli-review-v1',language=language,focus=focus,input_digest=identity,model_identity=self.identity,predicted=predicted,score=score,decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',reason='MODEL_CONTEXT_LIMIT' if predicted is None else 'MODEL_REVIEW_PROPOSAL' if predicted else 'NO_SIGNAL_DETECTED',review_requested=predicted!=0,score_is_calibrated=False,text_in_report=False,establishes_safety=False,permits_external_action=False,permits_notification=False))
        return results
