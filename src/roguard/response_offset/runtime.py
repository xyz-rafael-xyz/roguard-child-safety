"""Private frozen59 + draft-text offset prototype. No downloads or training.
One caller-owned reference encoder is reused. Installed closure is explicit;
full disk verification surrounds a batch, never claimed on every row.
"""
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path
import hashlib,json,math,types,struct,sys,importlib,threading
FORMAT='roguard-frozen59-text-offset-experiment-1'
LABELS=('appropriate','boundary','privacy','pressure','unsupported')
TRAINING_SPEC='647b586b6c883e2f1df91017d493007f42a598a9dacc34f6d466fe126d07627c'
RUNTIME_SPEC='a4a1afe83324c5f34ede2e4c8700fcd7213d9e48f4d607f1048e37ef059271c9'
PARENT_WEIGHTS='9206896b6b9025865f0579a954b913a186380edd936239114ac178145474b848'
LEXICAL_SOURCE='443e91dccae7fc2b6051790a58c4eac3755df5798de158086726d99c569366b5'
PROVENANCE=('training_plan_sha256','selection_freeze_sha256','fit_report_sha256','saved_fit_audit_sha256','native_feature_manifest_sha256')
DetailedResponse=namedtuple('DetailedResponse','response parent_logits logits route parent_identity artifact_identity input_digest')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def valid_sha(x):return type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def bounded_bytes(path,limit):
 with Path(path).open('rb') as f:raw=f.read(limit+1)
 require(len(raw)<=limit,'Artifact exceeds bounded reader');return raw
def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
def source_module(path,expected):
 raw=bounded_bytes(path,1024**2);require(sha_bytes(raw)==expected,'Lexical source changed');m=types.ModuleType('bound_offset_lexical');m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def adjusted_logits(parent_logits,draft,lexicon,theta,lexical):
 """Same two L2 text blocks and float64 addition order as the fixed fitter.
 Sparse scalar accumulation is portable; cached/native parity remains required.
 """
 import numpy as np
 require(type(parent_logits) in (tuple,list) and len(parent_logits)==5 and all(type(x) is float and math.isfinite(x) for x in parent_logits),'Raw native five logits')
 raw=np.asarray(parent_logits,np.float32);require(np.isfinite(raw).all() and all(struct.pack('>d',float(x))==struct.pack('>d',y) for x,y in zip(raw,parent_logits)),'Parent logits must preserve float32 including signed zero')
 width=sum(len(lexicon['channels'][kind]['vocabulary']) for kind in ('word','char'))
 require(theta.dtype==np.float64 and theta.shape==(5*(width+1),) and np.isfinite(theta).all(),'Offset coefficient shape/dtype')
 if not np.any(theta):return tuple(parent_logits)
 weights=theta[:5*width].reshape(5,width);bias=theta[5*width:];accum=np.zeros(5,np.float64);column=0;scale=1/math.sqrt(2)
 for kind in ('word','char'):
  for index,value in lexical.entries(draft,kind,lexicon['channels'][kind]):accum+=weights[:,column+index]*(value*scale)
  column+=len(lexicon['channels'][kind]['vocabulary'])
 result=raw.astype(np.float64)+accum+bias;require(np.isfinite(result).all(),'Nonfinite adjusted logits');return tuple(float(x) for x in result)

class Delegate:
 def __init__(self,readout,owner,draft,zero):self.original=readout;self.adapters=readout.adapters;self.owner=owner;self.draft=draft;self.zero=zero;self.raw=None;self.adjusted=None;self.calls=0
 def logits(self,features):
  values=self.original.logits(features);self.calls+=1;require(self.calls==1,'Exactly one native response readout')
  self.raw=tuple(values)
  if self.zero:
   # Same list object and arithmetic go to the unchanged parent softmax.
   adjusted_logits(values,self.draft,self.owner._lexicon,self.owner._theta,self.owner._lexical)
   self.adjusted=self.raw;return values
  self.adjusted=adjusted_logits(values,self.draft,self.owner._lexicon,self.owner._theta,self.owner._lexical);return list(self.adjusted)

class Frozen59TextOffset:
 def __init__(self,parent,closure,*,closure_type,native=None):
  import numpy as np
  if native is None:
   from roguard import gemma4_response_attention as native
  require(type(closure) is closure_type and closure_type.__name__=='Closure','Explicit reviewed closure type')
  closure_module=sys.modules.get(closure_type.__module__);require(closure_module is not None and getattr(closure_module,'Closure',None) is closure_type and Path(closure_module.__file__).resolve()==Path(__file__).with_name('closure.py').resolve(),'Actual sibling closure class/origin required')
  require(sha_bytes(bounded_bytes(closure_module.__file__,1024**2))==closure.asset_sha('closure_source'),'Actual closure source differs')
  self._native=native;self.parent=parent;self.closure=closure;self._batch=False;self._batch_owner=None
  require(closure.asset_sha('runtime_source')==sha_bytes(bounded_bytes(__file__,1024**2)) and closure.asset_sha('lexical_source')==LEXICAL_SOURCE,'Runtime/lexical source identity')
  require(closure.asset_sha('response59_spec')==RUNTIME_SPEC and closure.asset_sha('response59_training_manifest')==TRAINING_SPEC and closure.asset_sha('response59_weights')==PARENT_WEIGHTS,'Frozen59 identity')
  if closure._registration['scope']=='installed_candidate':
   require(native is importlib.import_module('roguard.gemma4_response_attention'),'Actual native module object required')
   require(type(parent) is native.Gemma4ResponseAttentionModel,'Actual native59 model required')
   package=Path(closure._registration['packages']['roguard']['root']).resolve();require(Path(native.__file__).resolve()==package/'gemma4_response_attention.py','Actual native import origin')
  self._parent_binding=(parent.identity,digest(parent._spec),parent._reference_binding)
  require(parent._spec['adapter_sha256']==PARENT_WEIGHTS and parent.identity==native.digest(dict(format=native.FORMAT,artifact=RUNTIME_SPEC,reference=parent.reference.identity)),'Parent specification/identity differs')
  require(Path(parent._weights).resolve()==closure.asset_path('response59_weights'),'Parent adapter path differs')
  if closure._registration['scope']=='installed_candidate':
   args=parent.reference._args
   require(Path(args[1]).resolve()==closure.asset_path('reference_weights') and args[2]==closure.asset_sha('reference_weights'),'Reference weights closure differs')
   require(canonical(json.loads(bounded_bytes(closure.asset_path('reference_spec'),1024**2)))==canonical(parent.reference._spec),'Reference specification closure differs')
   from roguard.gemma4_contract import FILES
   for name in FILES:require((Path(args[0])/name).resolve()==closure.asset_path('base:'+name),'Actual base/tokenizer path differs')
  manifest_raw=bounded_bytes(closure.asset_path('offset_manifest'),1024**2);require(sha_bytes(manifest_raw)==closure.asset_sha('offset_manifest'),'Manifest bytes differ');spec=json.loads(manifest_raw)
  required={'format','qualified','labels','parent_identity','reference_identity','parent59_runtime_spec_sha256','parent59_training_spec_sha256','parent59_adapter_sha256','lexical_source_sha256','runtime_source_sha256','closure_source_sha256','theta_sha256','lexicon_sha256','selected_arm','selected_candidate','query_sha256','probabilities_calibrated',*PROVENANCE}
  require(set(spec)==required and spec['format']==FORMAT and spec['qualified'] is False and spec['probabilities_calibrated'] is False and spec['labels']==list(LABELS) and spec['selected_arm']=='new_data' and spec['selected_candidate'] in ('zero','lambda0.001','lambda0.01'),'Offset manifest contract')
  for key,expected in [('parent_identity',parent.identity),('reference_identity',parent.reference.identity),('parent59_runtime_spec_sha256',RUNTIME_SPEC),('parent59_training_spec_sha256',TRAINING_SPEC),('parent59_adapter_sha256',PARENT_WEIGHTS),('lexical_source_sha256',LEXICAL_SOURCE),('runtime_source_sha256',closure.asset_sha('runtime_source')),('closure_source_sha256',closure.asset_sha('closure_source')),('theta_sha256',closure.asset_sha('offset_theta')),('lexicon_sha256',closure.asset_sha('offset_lexicon')),('query_sha256',native.QUERY_SHA256)]:require(spec[key]==expected,'Manifest identity differs: '+key)
  require(all(valid_sha(spec[k]) for k in PROVENANCE),'Research lineage digests required; files verified externally')
  self._lexical=source_module(closure.asset_path('lexical_source'),LEXICAL_SOURCE)
  lexraw=bounded_bytes(closure.asset_path('offset_lexicon'),32*1024**2);raw=bounded_bytes(closure.asset_path('offset_theta'),8*1024**2);require(sha_bytes(lexraw)==spec['lexicon_sha256'] and sha_bytes(raw)==spec['theta_sha256'],'Candidate bytes differ')
  import io
  lex=json.loads(lexraw);self._lexical.validate_lexicon(lex);theta=np.load(io.BytesIO(raw),allow_pickle=False);width=sum(len(lex['channels'][kind]['vocabulary']) for kind in ('word','char'))
  require(theta.dtype==np.float64 and theta.shape==(5*(width+1),) and np.isfinite(theta).all() and lex['train_rows']==13076,'Candidate dimensions/lexicon population')
  require(spec['selected_candidate']!='zero' or not np.any(theta),'Zero selection has nonzero coefficients')
  theta.flags.writeable=False;self._theta=theta;self._lexicon=lex;self._zero=not bool(np.any(theta));self._config=spec;self._configuration_hash=digest(spec);self._theta_identity=id(theta);self._lexical_identity=id(self._lexical)
  self.artifact_identity=digest(dict(format=FORMAT,manifest=closure.asset_sha('offset_manifest'),parent=parent.identity,closure=closure.identity));self.identity=parent.identity if self._zero else self.artifact_identity
  self._lexicon_hash=digest(lex);self._theta_hash=sha_bytes(theta.tobytes());self._settings_hash=digest(self._lexical.SETTINGS)
  self._readout_identity=id(parent._readout) if parent._readout is not None else None;self.closure.cheap_verify()
 def _guard(self):
  self.closure.cheap_verify()
  if self._readout_identity is not None:require(id(self.parent._readout)==self._readout_identity,'Parent readout replaced')
  require((self.parent.identity,digest(self.parent._spec),self.parent._reference_binding)==self._parent_binding and digest(self._config)==self._configuration_hash and id(self._theta)==self._theta_identity and not self._theta.flags.writeable and id(self._lexical)==self._lexical_identity and digest(self._lexical.SETTINGS)==self._settings_hash,'Runtime object/configuration changed')
 @contextmanager
 def batch(self):
  # Dispatch decisions and the complete verification lifetime use the same RLock.
  # A different thread cannot borrow this instance's pending verification.
  with self._native._LOCK:
   require(not self._batch,'Nested offset batches unsupported');self.closure.verify();self._guard();self._parameter_verify();self._batch=True;self._batch_owner=threading.get_ident();failure=None
   try:yield self
   except BaseException as exc:failure=exc;raise
   finally:
    self._batch=False;self._batch_owner=None
    try:self.closure.verify();self._guard();self._parameter_verify()
    except BaseException as closing:
     if failure is not None:raise BaseExceptionGroup('Inference and closing verification failed',[failure,closing]) from None
     raise
 def _parameter_verify(self):
  require(digest(self._lexicon)==self._lexicon_hash and sha_bytes(self._theta.tobytes())==self._theta_hash,'Loaded lexical parameters changed')
 def _validate(self,draft,language,context):
  require(language in ('ro','uk') and isinstance(draft,str) and 1<=len(draft.strip())<=6000 and len(draft)<=6000 and isinstance(context,str) and len(context)<=12000,'Expected bounded Romanian/Ukrainian draft and context');draft.encode('utf-8');context.encode('utf-8')
 def review_response(self,draft,language,context=''):
  with self._native._LOCK:
   if not self._batch:
    with self.batch():return self.review_response(draft,language,context)
   require(self._batch_owner==threading.get_ident(),'Foreign batch owner')
   self._validate(draft,language,context);self._guard()
   if self._zero:
    try:response=self.parent.review_response(draft,language,context)
    finally:
     if self._readout_identity is None and self.parent._readout is not None:self._readout_identity=id(self.parent._readout)
    self._guard();return response
   return self.review_response_detailed(draft,language,context).response
 def review_response_detailed(self,draft,language,context=''):
  native=self._native
  with native._LOCK:
   if not self._batch:
    with self.batch():return self.review_response_detailed(draft,language,context)
   require(self._batch_owner==threading.get_ident(),'Foreign batch owner')
   self._validate(draft,language,context);self._guard();parent=self.parent
   # Same lazy initialization as the unchanged native parent, not a second model.
   require((parent.reference.identity,parent.reference._args,native.digest(parent.reference._spec))==parent._reference_binding,'Native reference configuration changed')
   if parent.reference._engine is None:parent.reference._engine=native._Engine(*parent.reference._args)
   if parent._readout is None:parent._readout=native._AdapterReadout(parent._weights,parent._spec['adapter_sha256'],parent.reference._engine.model.base)
   original=parent._readout
   if self._readout_identity is None:self._readout_identity=id(original)
   require(id(original)==self._readout_identity,'Parent readout replaced')
   delegate=Delegate(original,self,draft,self._zero);parent._readout=delegate
   try:response=parent.review_response(draft,language,context)
   finally:parent._readout=original
   self._guard()
   if response['abstained']:
    require(delegate.calls==0 and delegate.raw is None,'Unsupported response unexpectedly produced logits');route='unsupported_parent';adjusted=None
   else:
    require(delegate.calls==1 and delegate.raw is not None,'Supported response missing same-call raw logits');route='zero_parent' if self._zero else 'text_offset';adjusted=delegate.adjusted
    if not self._zero:response=dict(response,model_identity=self.artifact_identity)
   return DetailedResponse(response,delegate.raw,adjusted,route,self.parent.identity,self.artifact_identity,response['input_digest'])
