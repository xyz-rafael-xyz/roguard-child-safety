"""Experimental native interpretation heads; explicit local assets only."""
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import threading

from .assurance import policy_digest
from .evidence import ONTOLOGY
from .native_review import NativeReviewModel, _TorchEngine, prepare_input
from .native_assets import PADDING_MULTIPLE

FIELDS=tuple(ONTOLOGY)
SIZES=tuple(len(ONTOLOGY[f]) for f in FIELDS)


def _number(v):
    # All fitted tensors use float32; reject overflow before loading the encoder.
    return type(v) in (int,float) and -3.4028234663852886e38<=v<=3.4028234663852886e38 and math.isfinite(v)


def _check_couplings(value):
    if value is None:return
    shapes={'scope':(3,3),'assertion_relation':(4,2)}
    if not isinstance(value,dict) or set(value)!=set(shapes):raise ValueError('Invalid coupling inventory')
    for name,(rows,columns) in shapes.items():
        matrix=value[name]
        if (not isinstance(matrix,list) or len(matrix)!=rows
                or any(not isinstance(row,list) or len(row)!=columns for row in matrix)
                or any(not _number(v) or v<=0 for row in matrix for v in row)):
            raise ValueError('Invalid coupling table')


def pair_probabilities(values,couplings):
    _check_couplings(couplings)
    if couplings is None:return None
    result={}
    for name,left,right in (('scope','subject','source'),('assertion_relation','assertion','relation')):
        matrix=[[x*y*couplings[name][i][j] for j,y in enumerate(values[right])] for i,x in enumerate(values[left])]
        total=sum(sum(row) for row in matrix)
        if not math.isfinite(total) or total<=0:raise ValueError('Invalid coupled normalization')
        result[name]=[[v/total for v in row] for row in matrix]
    return result


def factorized_review_mass(values,*,couplings=None):
    """Closed form of the binary policy under the declared categorical factors.

    Time affects the action subtype, but not the binary need for review. The
    complete 432-world policy is checked against this expression in tests.
    """
    if not isinstance(values,dict) or set(values)!=set(FIELDS):raise ValueError('Six field vectors required')
    for field,size in zip(FIELDS,SIZES):
        p=values[field]
        if (not isinstance(p,list) or len(p)!=size or any(not _number(v) or not 0<=v<=1 for v in p)
                or abs(sum(p)-1)>1e-6):raise ValueError('Invalid categorical probabilities')
    def p(field,label):return values[field][ONTOLOGY[field].index(label)]
    if couplings is not None:
        pairs=pair_probabilities(values,couplings);scope=pairs['scope'];ar=pairs['assertion_relation']
        sf=ONTOLOGY['subject'].index('FICTIONAL');fi=ONTOLOGY['source'].index('FICTION')
        neg=ONTOLOGY['assertion'].index('NEGATED');ret=ONTOLOGY['assertion'].index('RETRACTED');new=ONTOLOGY['relation'].index('NEW')
        mismatch=sum(scope[i][j] for i in range(3) for j in range(3) if (i==sf)!=(j==fi))
        real=sum(scope[i][j] for i in range(3) for j in range(3) if i!=sf and j!=fi)
        fiction_review=1-sum(ar[i][new] for i in range(4) if i!=ret)
        return max(0.,min(1.,p('concern','1')*(mismatch+scope[sf][fi]*fiction_review+real*(1-ar[neg][new]))))
    fictional=p('subject','FICTIONAL');fiction=p('source','FICTION')
    negated=p('assertion','NEGATED');retracted=p('assertion','RETRACTED');retracts=p('relation','RETRACTS')
    mismatch=fictional*(1-fiction)+(1-fictional)*fiction
    fiction_review=retracted+retracts-retracted*retracts
    real_review=1-(1-retracts)*negated
    mass=p('concern','1')*(mismatch+fictional*fiction*fiction_review+(1-fictional)*(1-fiction)*real_review)
    return max(0.,min(1.,mass))


def _network(torch,hidden):
    if hidden==0:return torch.nn.Linear(1536,sum(SIZES))
    return torch.nn.Sequential(torch.nn.Linear(1536,hidden),torch.nn.GELU(),
                               torch.nn.Dropout(.1),torch.nn.Linear(hidden,sum(SIZES)))


def _native_states(engine,inputs,*,include_logits=False):
    """Retain the final layer only; optional logits reproduce the native head."""
    outputs=engine.model.base_model(**inputs,return_dict=True,output_hidden_states=False)
    hidden=outputs.last_hidden_state.float()
    logits=None
    if include_logits:
        if engine.model.config.model_type=='bert':
            logits=engine.model.classifier(engine.model.dropout(outputs.pooler_output))
        elif engine.model.config.model_type=='roberta':
            logits=engine.model.classifier(hidden)
        else:raise ValueError('Unsupported native classifier architecture')
    return hidden,logits


class _Engine:
    def __init__(self,native,spec):
        import numpy as np
        self.np=np;self.native=_TorchEngine(*native._engine_args)
        torch=self.native.torch
        if not self.native.tokenizer.is_fast:raise ValueError('Focal sequence identities require a fast tokenizer')
        self.head=_network(torch,spec['hidden']).float().eval()
        tensors={k:torch.tensor(v,dtype=torch.float32) for k,v in spec['weights'].items()}
        if any(not torch.isfinite(v).all() for v in tensors.values()):raise ValueError('Nonfinite native head weights')
        self.head.load_state_dict(tensors,strict=True)
        self.mean=np.asarray(spec['mean'],dtype=np.float32);self.scale=np.asarray(spec['scale'],dtype=np.float32)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale<=0).any():
            raise ValueError('Invalid float32 feature normalization')

    def predict(self,pairs):
        engine=self.native;torch=engine.torch;np=self.np
        encoded=engine.tokenizer([p[0] for p in pairs],[p[1] for p in pairs],padding=False,truncation=False)
        valid=[i for i,ids in enumerate(encoded['input_ids']) if len(ids)<=512]
        result=[None]*len(pairs)
        with torch.inference_mode():
            for offset in range(0,len(valid),8):
                ix=valid[offset:offset+8]
                batch=engine.tokenizer.pad([{k:v[i] for k,v in encoded.items()} for i in ix],
                    return_tensors='pt',pad_to_multiple_of=PADDING_MULTIPLE)
                mask=np.zeros(tuple(batch['input_ids'].shape),dtype=np.float32)
                for j,i in enumerate(ix):
                    ids=encoded.sequence_ids(i);mask[j,:len(ids)]=[v==1 for v in ids]
                if (mask.sum(axis=1)==0).any():raise ValueError('Focal token span is empty')
                hidden,_=_native_states(engine,{k:v.to(engine.device) for k,v in batch.items()})
                focal=torch.from_numpy(mask).to(engine.device)
                pooled=(hidden*focal[:,:,None]).sum(dim=1)/focal.sum(dim=1)[:,None]
                features=torch.cat((hidden[:,0,:],pooled),dim=1).cpu().numpy()
                features=np.ascontiguousarray((features-self.mean)/self.scale,dtype=np.float32)
                if features.shape!=(len(ix),1536) or not np.isfinite(features).all():raise ValueError('Invalid native representations')
                logits=self.head(torch.from_numpy(features))
                vectors=[torch.softmax(v,dim=-1).tolist() for v in torch.split(logits,SIZES,dim=-1)]
                for j,i in enumerate(ix):result[i]={field:values[j] for field,values in zip(FIELDS,vectors)}
                del hidden,focal,pooled
        if engine.device=='mps':torch.mps.empty_cache()
        return result


class NativeFieldsModel:
    """Small semantic heads over a pinned native checkpoint; no default selection."""
    def __init__(self,heads,native_artifact,checkpoint,base,manifest,*,expected_heads_sha256,device='cpu'):
        with Path(heads).open('rb') as stream:raw=stream.read(8_000_001)
        if (len(raw)>8_000_000 or not isinstance(expected_heads_sha256,str)
                or hashlib.sha256(raw).hexdigest()!=expected_heads_sha256):raise ValueError('Native head artifact identity or size differs')
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:unpacked=stream.read(32_000_001)
        if len(unpacked)>32_000_000:raise ValueError('Native head artifact is too large')
        spec=json.loads(unpacked)
        required={'format','language','hidden','dimensions','native_artifact_sha256','mean','scale','weights','threshold','fields','sizes','distribution','policy_digest','couplings'}
        if (not isinstance(spec,dict) or set(spec)!=required or spec['format']!='roguard-native-fields-experiment-v1'
                or spec['language'] not in ('ro','uk') or type(spec['hidden']) is not int or spec['hidden'] not in (0,256)
                or type(spec['dimensions']) is not int or spec['dimensions']!=1536 or spec['fields']!=list(FIELDS)
                or spec['sizes']!=list(SIZES) or spec['distribution']!=('independent categorical fields; uncalibrated' if spec['couplings'] is None else 'pair-coupled categorical fields; uncalibrated')
                or spec['policy_digest']!=policy_digest()):raise ValueError('Unsupported native head contract')
        _check_couplings(spec['couplings'])
        for name in ('mean','scale'):
            if not isinstance(spec[name],list) or len(spec[name])!=1536 or any(not _number(v) for v in spec[name]):
                raise ValueError('Invalid feature normalization')
        if (any(v<1e-6 for v in spec['scale']) or not _number(spec['threshold']) or not 0<=spec['threshold']<=1
                or not isinstance(spec['weights'],dict)):raise ValueError('Invalid head parameters')
        shapes=({'weight':(sum(SIZES),1536),'bias':(sum(SIZES),)} if spec['hidden']==0 else
                {'0.weight':(256,1536),'0.bias':(256,),'3.weight':(sum(SIZES),256),'3.bias':(sum(SIZES),)})
        if set(spec['weights'])!=set(shapes):raise ValueError('Head tensor inventory differs')
        for name,shape in shapes.items():
            value=spec['weights'][name]
            if not isinstance(value,list) or len(value)!=shape[0]:raise ValueError('Head tensor shape differs')
            if len(shape)==2:
                if any(not isinstance(row,list) or len(row)!=shape[1] for row in value):raise ValueError('Head tensor shape differs')
                value=[v for row in value for v in row]
            if any(not _number(v) for v in value):raise ValueError('Invalid head tensor values')
        native=NativeReviewModel(native_artifact,checkpoint,base,manifest,
            expected_artifact_sha256=spec['native_artifact_sha256'],device=device)
        if native.language!=spec['language']:raise ValueError('Native/head languages differ')
        self._native=native;self._spec=spec;self._engine=None;self._lock=threading.RLock();self.identity=expected_heads_sha256

    def predict(self,payload,*,focus=None):return self.predict_many([payload],focuses=[focus])[0]

    def predict_many(self,payloads,*,focuses=None):
        if not isinstance(payloads,list) or not 1<=len(payloads)<=128:raise ValueError('Expected one to128 conversations')
        if focuses is None:focuses=[None]*len(payloads)
        if not isinstance(focuses,list) or len(focuses)!=len(payloads):raise ValueError('Focus count differs')
        prepared=[prepare_input(payload,focus) for payload,focus in zip(payloads,focuses)]
        if any(p['language']!=self._spec['language'] for p in payloads):raise ValueError('Language differs from selected model')
        with self._lock:
            if self._engine is None:self._engine=_Engine(self._native,self._spec)
            values=self._engine.predict([p[0] for p in prepared])
        if len(values)!=len(payloads):raise ValueError('Incomplete semantic head output')
        results=[]
        for p,(_,focus,identity) in zip(values,prepared):
            score=None if p is None else factorized_review_mass(p,couplings=self._spec['couplings'])
            predicted=None if score is None else int(score>=self._spec['threshold'])
            interpretation=None if p is None else {f:ONTOLOGY[f][max(range(len(p[f])),key=p[f].__getitem__)] for f in FIELDS}
            results.append(dict(format='roguard-native-fields-v1',language=self._spec['language'],focus=focus,input_digest=identity,
                model_identity=self.identity,native_model_identity=self._native.identity,policy_digest=self._spec['policy_digest'],
                predicted=predicted,score=score,interpretation=interpretation,field_probabilities=p,
                interpretation_is_head_argmax=True,
                decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',
                reason='NATIVE_CONTEXT_LIMIT' if predicted is None else 'NATIVE_FIELDS_REVIEW_PROPOSAL',review_requested=predicted!=0,
                pair_probabilities=None if p is None else pair_probabilities(p,self._spec['couplings']),
                distribution=self._spec['distribution'],
                probabilities_are_calibrated=False,field_independence_assumed=self._spec['couplings'] is None,
                factor_independence_assumed=True,text_in_report=False,
                establishes_safety=False,permits_external_action=False,permits_notification=False))
        return results
