"""Explicit local Qwen review experiments; never grants external permissions."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading

from .interpreter import prepare_input
from .native_review import _read, _digest, _sha
from .qwen_contract import POLICY, BASE_MODEL, BASE_REVISION, FILES

_LOCK = threading.RLock()
CONTRACT = hashlib.sha256(POLICY.encode()).hexdigest()
ADAPTER_CONFIG = dict(fine_tune_type='lora',num_layers=1,
                      lora_parameters=dict(rank=8,dropout=0.0,scale=16.0))


def _snapshot(base, destination):
    destination.mkdir()
    for name, expected in FILES.items():
        digest=hashlib.sha256();total=0
        with (Path(base)/name).open('rb') as source, (destination/name).open('xb') as target:
            while True:
                chunk=source.read(1024*1024)
                if not chunk:break
                total+=len(chunk)
                if total>expected['bytes']:raise ValueError('Qwen base asset exceeds pinned size')
                digest.update(chunk);target.write(chunk)
        if total!=expected['bytes'] or digest.hexdigest()!=expected['sha256']:
            raise ValueError('Qwen base asset identity differs: '+name)


class _Engine:
    def __init__(self, base, adapter, digest):
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
            from safetensors.numpy import load as load_tensors
            import numpy as np
        except ImportError as error:
            raise ImportError('Qwen review requires Apple Silicon and the mlx extra') from error
        raw=_read(adapter,8_000_000)
        if _sha(raw)!=digest:raise ValueError('Qwen adapter identity differs')
        tensors=load_tensors(raw)
        shapes={}
        for name,inputs,outputs in [('mlp.down_proj',12288,4096),('mlp.gate_proj',4096,12288),
                ('mlp.up_proj',4096,12288),('self_attn.k_proj',4096,1024),
                ('self_attn.o_proj',4096,4096),('self_attn.q_proj',4096,8192),('self_attn.v_proj',4096,1024)]:
            prefix='language_model.model.layers.31.'+name
            shapes[prefix+'.lora_a']=(inputs,8);shapes[prefix+'.lora_b']=(8,outputs)
        if set(tensors)!=set(shapes) or any(v.shape!=shapes[k] or v.dtype!=np.float32 or not np.isfinite(v).all()
                              for k,v in tensors.items()):
            raise ValueError('Invalid Qwen adapter tensors')
        del tensors
        self.mx=mx;self.make_cache=make_prompt_cache
        with tempfile.TemporaryDirectory(prefix='roguard-qwen-') as temporary:
            directory=Path(temporary);_snapshot(base,directory/'base')
            assets=directory/'adapter';assets.mkdir()
            (assets/'adapters.safetensors').write_bytes(raw)
            (assets/'adapter_config.json').write_text(json.dumps(ADAPTER_CONFIG))
            self.model,self.tokenizer=load(str(directory/'base'),adapter_path=str(assets),
                tokenizer_config={'trust_remote_code':False},lazy=False)
            self.model.set_dtype(mx.float32);self.model.eval();mx.eval(self.model.parameters())
        self.text=self.model.language_model
        ids=[self.tokenizer.encode(str(i),add_special_tokens=False) for i in (0,1)]
        if any(len(v)!=1 for v in ids):raise ValueError('Decision digits are not single tokens')
        self.ids=mx.array([v[0] for v in ids]);self.valid_ids={v[0] for v in ids}

    def tokens(self, prefix):
        payload=dict(language=prefix['language'],prior_turns=prefix['turns'][:-1],
                     focal_message=prefix['turns'][-1])
        prompt=self.tokenizer.apply_chat_template(
            [dict(role='system',content=POLICY),dict(role='user',content=json.dumps(payload,ensure_ascii=False))],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        return self.tokenizer.encode(prompt,add_special_tokens=False)

    def scores(self, prefixes):
        mx=self.mx
        sequences=[self.tokens(p) for p in prefixes]
        valid=[s for s in sequences if len(s)<=2048]
        common=0
        if valid:
            for values in zip(*valid):
                if len(set(values))!=1:break
                common+=1
            common=min(common,min(map(len,valid))-1)
        # This cache belongs only to this call and is discarded afterwards.
        cache=self.make_cache(self.model)
        if common:mx.eval(self.text.model(mx.array([valid[0][:common]]),cache=cache))
        output=[]
        for sequence in sequences:
            if len(sequence)>2048:
                output.append((None,False));continue
            hidden=self.text.model(mx.array([sequence[common:]]),cache=deepcopy(cache))[:,-1:,:]
            logits=(self.text.model.embed_tokens.as_linear(hidden) if self.text.args.tie_word_embeddings
                    else self.text.lm_head(hidden))[0,-1].astype(mx.float32)
            score=float(mx.softmax(logits[self.ids])[1].item())
            valid_token=int(mx.argmax(logits).item()) in self.valid_ids
            output.append((score,valid_token))
        return output


class QwenReviewModel:
    """Hash-pinned experimental adapter, loaded only from local assets."""
    def __init__(self,artifact,adapter,base,*,expected_sha256):
        raw=_read(artifact,65_536)
        if _sha(raw)!=_digest(expected_sha256):raise ValueError('Qwen artifact identity differs')
        spec=json.loads(raw)
        required={'format','base_model','base_revision','policy_sha256','adapter_sha256','thresholds'}
        if not isinstance(spec,dict) or set(spec)!=required or spec['format']!='roguard-qwen-review-experiment-v1':
            raise ValueError('Unsupported Qwen artifact')
        if (spec['base_model'],spec['base_revision'],spec['policy_sha256'])!=(BASE_MODEL,BASE_REVISION,CONTRACT):
            raise ValueError('Qwen model or policy contract differs')
        limits=spec['thresholds']
        if not isinstance(limits,dict) or set(limits)!={'ro','uk'} or any(
            type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in limits.values()):
            raise ValueError('Invalid Qwen language thresholds')
        self._thresholds=dict(limits);self._identity=expected_sha256;self._engine=None
        self._args=(Path(base),Path(adapter),_digest(spec['adapter_sha256']))

    @classmethod
    def from_bundle(cls,path,*,expected_sha256):
        from .field_pipeline import _local_path
        path=Path(path).resolve();raw=_read(path,65_536)
        if _sha(raw)!=_digest(expected_sha256):raise ValueError('Qwen bundle identity differs')
        spec=json.loads(raw)
        if (not isinstance(spec,dict) or set(spec)!={'format','artifact_sha256','assets'}
                or spec['format']!='roguard-qwen-review-bundle-v1'
                or not isinstance(spec['assets'],dict)
                or set(spec['assets'])!={'artifact','adapter','base'}):
            raise ValueError('Unsupported Qwen bundle')
        assets={key:_local_path(path.parent,value) for key,value in spec['assets'].items()}
        return cls(**assets,expected_sha256=spec['artifact_sha256'])

    @property
    def identity(self):return self._identity

    def predict(self,payload,*,focus=None):
        return self.predict_many([payload],focuses=[focus])[0]

    def predict_many(self,payloads,*,focuses=None):
        if not isinstance(payloads,list) or not 1<=len(payloads)<=128:
            raise ValueError('Expected one to 128 conversations')
        if focuses is None:focuses=[None]*len(payloads)
        if not isinstance(focuses,list) or len(focuses)!=len(payloads):
            raise ValueError('Focal indices do not match the batch')
        prepared=[prepare_input(p,f) for p,f in zip(payloads,focuses)]
        with _LOCK:
            if self._engine is None:self._engine=_Engine(*self._args)
            scores=self._engine.scores([p[0] for p in prepared])
        if not isinstance(scores,list) or len(scores)!=len(prepared):
            raise ValueError('Incomplete Qwen output')
        results=[]
        for (score,valid),(prefix,focus,identity) in zip(scores,prepared):
            if type(valid) is not bool or (score is not None and
                (type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1)):
                raise ValueError('Invalid Qwen score or validity')
            if valid and score is None:raise ValueError('Missing valid Qwen score')
            predicted=int(score>=self._thresholds[prefix['language']]) if valid else None
            results.append(dict(format='roguard-qwen-review-v1',language=prefix['language'],focus=focus,
                input_digest=identity,model_identity=self.identity,score=score,predicted=predicted,
                valid_label_token=valid,decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',
                reason='INVALID_OR_OVERSIZED_MODEL_OUTPUT' if predicted is None else 'MODEL_REVIEW_PROPOSAL' if predicted else 'NO_SIGNAL_DETECTED',
                review_requested=predicted!=0,score_is_calibrated=False,text_in_report=False,
                establishes_safety=False,permits_external_action=False,permits_notification=False))
        return results
