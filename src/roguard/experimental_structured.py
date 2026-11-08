"""Private structured-route prototype. No loader, calibration fit or downloads.
A caller supplies one reviewed language component and a complete bilingual closure.
Sequential worker process exit is the caller's one-backbone resource contract.
This wrapper neither loads models nor enforces a global model count.
"""
from copy import deepcopy
from pathlib import Path,PurePosixPath
import gzip,hashlib,json,math,sys,platform,importlib,importlib.metadata,importlib.util,itertools
from roguard.evidence import ONTOLOGY,_complete_action
from roguard.assurance import digest,policy_digest
from roguard.trajectory_data import validate_conversation,prefix,input_identity
from roguard.trajectory_memory import units,verify_view,MEMORY_CONTRACT
from roguard.trajectory_neural import prepare_memory,factors_from_logits
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_sets import WORLDS,predict_set
from roguard.gemma4_review import _score
from roguard.native_fields import pair_probabilities
from roguard.gemma4_contract import FILES as BASE_FILES,VERSIONS as RUNTIME_VERSIONS
SPEC75={'ro':'5fdeae97843d873755ed41a02453b5c29823fd8028508c3e99d1586175c8f4a9','uk':'bbf3fea8ee298c40d86701851fe045352ea1a924ba5287dd52b70e59ce25de27'}
SPEC9={'ro':'6b2ea2979ffa43c735d8f8a6e3f7e1fbf6cf5c085889e8d484a13bf43026ccc5','uk':'fc86fed9981b786149c4f6855c3d98f07067654cf2af05d40aef589345be34ed'}
PIN75W={'ro':'1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3','uk':'466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}
PIN9W={'ro':'114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09','uk':'0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f'}
SELECTOR_SHA='81493f522678301dfc3196345cd18a577ad39d3b4cc386260cd10866d85a6eea'
MASS_ATOL=2e-14

def require(x,m):
 if not x:raise ValueError(m)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def read(p):return json.loads(Path(p).read_bytes())
def file_record(p):return dict(bytes=Path(p).stat().st_size,sha256=sha(p))
def python_inventory(root):
 root=Path(root);require(root.is_dir() and not root.is_symlink(),'Regular package root required');result={}
 for p in sorted(root.rglob('*')):
  require(not p.is_symlink(),'Package symlinks are not admitted')
  if p.is_file() and p.suffix=='.py':result[p.relative_to(root).as_posix()]=file_record(p)
 require(result,'Empty Python source inventory');return result

def host_diagnostics():
 # No hostname, user, serial, installation path or other machine identifier.
 return dict(system=platform.system(),release=platform.release(),machine_architecture=platform.machine(),python_version=platform.python_version(),python_implementation=platform.python_implementation())

class Closure:
 """Verify actual files, exact Python inventories and imported module origins.
 The registration is trusted input; its hash must be bound by the calling study.
 """
 def __init__(self,registration_path,expected_sha256):
  self._path=Path(registration_path);self._sha=expected_sha256;require(sha(self._path)==expected_sha256,'Closure registration changed');self._registration=read(self._path);self._registration_digest=digest(self._registration)
  r=self._registration;require(r['format']=='roguard-structured-installed-closure-1' and r['qualified'] is False and r['scope'] in ('synthetic_fixture','installed_candidate'),'Invalid closure scope')
  require(isinstance(r['packages'],dict) and r['packages'] and isinstance(r['assets'],dict) and r['assets'],'Explicit packages and assets required')
  require(set(r['assets'])==set(r['required_asset_ids']),'Exact transitive asset inventory required')
  if r['scope']=='installed_candidate':
   require(r['registered'] is True and {'roguard','mlx_lm','mlx','numpy','transformers','tokenizers','huggingface_hub','safetensors'}<=set(r['packages']),'Registered installed runtime/dependency roots required')
   versions={p['distribution']:p['version'] for p in r['packages'].values()}
   require(all(versions.get(name)==version for name,version in RUNTIME_VERSIONS.items()),'Pinned runtime dependency versions required')
   require(set(r['required_runtime_assets'])<=set(r['assets']) and r['native_dependencies'],'Runtime and native dependency closure required')
  self.verify();self.receipt=self._receipt();self.identity=digest(self.receipt)
 def _receipt(self):
  r=self._registration
  return dict(format='roguard-verified-structured-closure-receipt-1',scope=r['scope'],packages={name:{'files':v['files'],'distribution':v.get('distribution'),'version':v.get('version')} for name,v in r['packages'].items()},assets={name:{k:v[k] for k in ('bytes','sha256')} for name,v in r['assets'].items()},asset_edges=r['asset_edges'],native_dependencies={name:{k:v[k] for k in ('bytes','sha256')} for name,v in r['native_dependencies'].items()},declared_supported_runtime_contract=r['declared_supported_runtime_contract'])
 def asset_path(self,name):return Path(self._registration['assets'][name]['path'])
 def asset_sha(self,name):return self._registration['assets'][name]['sha256']
 def cheap_verify(self):
  require(digest(self._registration)==self._registration_digest,'Closure configuration mutated')
  if hasattr(self,'identity'):require(digest(self._receipt())==self.identity,'Closure receipt mutated')
  for name,p in self._registration['packages'].items():
   root=Path(p['root']).resolve()
   for modname,module in list(sys.modules.items()):
    if modname==name or modname.startswith(name+'.'):
     path=getattr(module,'__file__',None)
     if path is None:require({Path(v).resolve() for v in getattr(module,'__path__',[])}=={root},'Unbound namespace import')
     else:
      actual=Path(path).resolve();require(actual.is_relative_to(root),'Module imported outside frozen package')
      require(actual.relative_to(root).as_posix() in p['files'] or any(Path(v['path']).resolve()==actual for v in self._registration['native_dependencies'].values()),'Uninventoried imported module origin')
  return True
 def verify(self):
  self.cheap_verify();require(sha(self._path)==self._sha,'Closure registration changed');r=self._registration
  for name,p in r['packages'].items():
   root=Path(p['root']).resolve();require(canonical(python_inventory(root))==canonical(p['files']),'Python source inventory differs: '+name)
   if r['scope']=='installed_candidate':
    distribution=importlib.metadata.distribution(p['distribution']);require(distribution.version==p['version'],'Installed dependency version differs')
    require(root.is_relative_to(Path(distribution.locate_file('')).resolve()),'Package outside its installed distribution')
    found=importlib.util.find_spec(name);require(found is not None and (found.submodule_search_locations is not None) and {Path(x).resolve() for x in found.submodule_search_locations}=={root},'Imported package root differs')
   for modname,module in list(sys.modules.items()):
    if modname==name or modname.startswith(name+'.'):
     path=getattr(module,'__file__',None)
     if path is None:
      require({Path(v).resolve() for v in getattr(module,'__path__',[])}=={root},'Unbound namespace import')
     else:
      actual=Path(path).resolve();require(actual.is_relative_to(root),'Module imported outside frozen package')
      rel=actual.relative_to(root).as_posix()
      if rel in p['files']:require(canonical(file_record(actual))==canonical(p['files'][rel]),'Changed imported Python module')
      else:require(any(Path(b['path']).resolve()==actual and canonical(file_record(actual))==canonical({k:b[k] for k in ('bytes','sha256')}) for b in r['native_dependencies'].values()),'Uninventoried imported native module')
  for group in ('assets','native_dependencies'):
   for name,b in r[group].items():
    path=Path(b['path']);require(path.is_file() and not path.is_symlink() and canonical(file_record(path))==canonical({k:b[k] for k in ('bytes','sha256')}),'Asset bytes differ: '+name)
  for edge in r['asset_edges']:
   value=read(self.asset_path(edge['from']))
   for part in edge['keys']:value=value[part]
   require(type(value) is str and value==self.asset_sha(edge['to']),'Transitive manifest digest differs')
  if hasattr(self,'identity'):require(digest(self._receipt())==self.identity,'Closure object mutated')
  return True


def coupled_worlds(logits,spec):
 factors={};offset=0
 for field,values in ONTOLOGY.items():
  raw=logits[offset:offset+len(values)];top=max(raw);p=[math.exp(x-top) for x in raw];total=math.fsum(p);factors[field]=[x/total for x in p];offset+=len(values)
 pairs=pair_probabilities(factors,spec['couplings']);indices=[{v:i for i,v in enumerate(values)} for values in ONTOLOGY.values()]
 result=[]
 for world in WORLDS:
  s,t,a,src,rel,c=[ix[world[k]] for k,ix in zip(ONTOLOGY,indices)]
  result.append(pairs['scope'][s][src]*pairs['assertion_relation'][a][rel]*factors['time'][t]*factors['concern'][c])
 total=math.fsum(result);require(abs(total-1)<=MASS_ATOL,'World normalization differs');return [v/total for v in result]


def check_evidence(original,evidence):
 previous=None
 for span in evidence:
  require(set(span)=={'turn','start','end'} and all(type(v) is int for v in span.values()),'Invalid evidence schema')
  key=span['turn'],span['start'],span['end'];require(previous is None or key>previous,'Evidence repeated or unordered');previous=key
  require(0<=key[0]<len(original['turns']) and 0<=key[1]<key[2]<=len(original['turns'][key[0]]['text']),'Evidence outside original prefix')


def rendering_segments(original,view,inference):
 """Map exact rendered substrings; inserted join spaces are not source evidence."""
 allunits={};selected={}
 for s in units(original):allunits.setdefault(s['turn'],set()).add((s['start'],s['end']))
 for s in view['spans']:selected.setdefault(s['turn'],[]).append(s)
 require(len(selected)==len(inference['turns']),'Rendered turn count differs');result=[]
 for out_index,source_index in enumerate(sorted(selected)):
  spans=selected[source_index];rendered=inference['turns'][out_index];source=original['turns'][source_index]
  require(rendered['role']==source['role'],'Rendered role changed')
  if {(s['start'],s['end']) for s in spans}==allunits[source_index]:
   require(rendered['text']==source['text'],'Complete turn bytes changed');result.append(dict(inference_turn=out_index,inference_start=0,inference_end=len(source['text']),source_turn=source_index,source_start=0,source_end=len(source['text'])))
  else:
   require(rendered['text']==' '.join(s['text'] for s in spans),'Partial rendering differs');offset=0
   for s in spans:
    require(s['text']==source['text'][s['start']:s['end']],'Original Unicode substring changed');result.append(dict(inference_turn=out_index,inference_start=offset,inference_end=offset+len(s['text']),source_turn=source_index,source_start=s['start'],source_end=s['end']));offset+=len(s['text'])+1
 return result

class InactiveLanguageError(ValueError):
 """Configuration error; never an unsupported-input or fallback response."""

class StructuredWrapper:
 def __init__(self,components,selector,closure,*,active_language,mode='direct-first',budget=1200):
  require(type(active_language) is str and active_language in ('ro','uk'),'Explicit active language required')
  require(type(components) is dict and set(components)=={active_language} and all(callable(getattr(c,'predict_detailed',None)) for c in components.values()),'Exactly one matching detailed component required')
  self._active_language=active_language
  require(mode in ('direct-first','uniform') and type(budget) is int and budget==1200,'Frozen explicit mode/budget required')
  require(isinstance(closure,Closure),'Verified closure object required');closure.verify();self.closure=closure;self.components=dict(components);self.selector=deepcopy(selector);self.mode=mode;self.budget=budget
  self.specs={};self.manifest_ids={};self.manifest_values={}
  if closure._registration['scope']=='installed_candidate':
   require(closure.asset_sha('selector')==SELECTOR_SHA,'Frozen1200selector differs')
   require(type(self.selector) is TrajectoryModel,'Reviewed memory selector type required')
   require(closure.asset_path('selector').stat().st_size<=20_000_000,'Selector artifact exceeds bound')
   with gzip.open(closure.asset_path('selector'),'rb') as source:selector_bytes=source.read(40_000_001)
   require(len(selector_bytes)<=40_000_000 and canonical(json.loads(selector_bytes))==canonical(self.selector.spec),'Loaded selector specification differs')
   for lang in ('ro','uk'):
    require(closure.asset_sha(lang+'_weights75')==PIN75W[lang] and closure.asset_sha(lang+'_adapter9')==PIN9W[lang],'Parent weight closure differs')
    manifest=read(closure.asset_path(lang+'_residual_manifest'));require(manifest['language']==lang and manifest['tensor_sha256']==closure.asset_sha(lang+'_residual_tensors'),'Residual asset closure differs')
    packagefiles=closure._registration['packages']['roguard']['files']
    require(manifest['runtime_source_sha256']==packagefiles['experimental_role_residual.py']['sha256'] and manifest['alignment_source_sha256']==packagefiles['role_alignment.py']['sha256'],'Installed residual source closure differs')
    expected_parent={'original_v9_adapter_sha256':PIN9W[lang],'original_v9_spec_sha256':SPEC9[lang],'head75_source_sha256':PIN75W[lang],'spec_sha256':SPEC75[lang]}
    require(all(manifest['parent'].get(k)==v for k,v in expected_parent.items()),'Residual parent asset closure differs')
    for name,key in [('selected_checkpoint','selected_checkpoint_sha256'),('fit_plan','training_plan_sha256'),('feature_manifest','feature_manifest_sha256'),('selection_proof','selection_proof_sha256')]:require(manifest['provenance'][key]==closure.asset_sha(lang+'_'+name),'Selected provenance closure differs')
   for name,b in BASE_FILES.items():require(closure.asset_sha('base:'+name)==b['sha256'] and closure._registration['assets']['base:'+name]['bytes']==b['bytes'],'Pinned base asset closure differs')
  for lang in ('ro','uk'):
   self.specs[lang]={}
   for name,pins in [('75',SPEC75),('9',SPEC9)]:
    key=lang+'_spec'+name;require(closure.asset_sha(key)==pins[lang],'Original specification pin differs');spec=read(closure.asset_path(key));require(spec['language']==lang and spec['policy_digest']==policy_digest(),'Spec language/policy differs');self.specs[lang][name]=spec
   require(self.specs[lang]['75']['mode']=='coupled_policy' and self.specs[lang]['9']['mode']==('policy' if lang=='ro' else 'binary'),'Frozen route mode differs')
   self.manifest_ids[lang]=closure.asset_sha(lang+'_residual_manifest');self.manifest_values[lang]=read(closure.asset_path(lang+'_residual_manifest'))
  require(getattr(self.selector,'identity',None)==closure.asset_sha('selector'),'Selector identity differs');self.selector_state=digest(self.selector.spec);self.spec_state=digest(self.specs);self.manifest_state=digest(self.manifest_values)
  self.contract=dict(format='roguard-full-route-structured-pipeline-1',closure=closure.receipt,wrapper_source_sha256=sha(__file__),ontology={k:list(v) for k,v in ONTOLOGY.items()},policy_digest=policy_digest(),mode=mode,budget=budget,memory_contract=MEMORY_CONTRACT,preserve_complete=True,selector_sha256=self.selector.identity,distribution_contracts={'residual':'coupled-scope-assertion-relation-time-concern-worlds-v1','original_v9:ro':'independent-six-fields-v1;policy-score','original_v9:uk':'independent-six-fields-v1;separate-legacy-binary-score','unsupported':'full432-no-confidence'},residual_maximum_tokens=613,original_v9_maximum_tokens=1024,maximum_direct_turns=12)
  self.identity=digest(self.contract);self._active_batch=None;self._component_bindings=self._component_identity();self._verify_installed_components()
 @property
 def active_language(self):return self._active_language
 def _require_active(self,payload):
  require(type(payload) is dict and payload.get('language') in ('ro','uk'),'Valid payload language required')
  if payload['language']!=self.active_language:raise InactiveLanguageError('Language is not loaded in this worker: '+payload['language'])
 def _verify_installed_components(self):
  if self.closure._registration['scope']!='installed_candidate':return
  require(type(self.selector) is TrajectoryModel and self.selector.identity==self.closure.asset_sha('selector') and getattr(self.selector.relevance,'__func__',None) is TrajectoryModel.relevance,'Loaded selector binding differs')
  runtime=importlib.import_module('roguard.experimental_role_residual')
  for lang,component in self.components.items():
   require(type(component) is runtime.ExperimentalRoleResidualModel and component.enabled is True and getattr(component.predict_detailed,'__func__',None) is runtime.ExperimentalRoleResidualModel.predict_detailed,'Reviewed enabled model type required')
   engine=component._parent;require(type(engine) is runtime.OptionalResidualEngine and engine.enabled is True and type(engine.residual) is runtime.ResidualArrays,'Reviewed enabled residual engine required')
   require(engine.residual.identity==self.manifest_ids[lang] and canonical(engine.residual.manifest)==canonical(self.manifest_values[lang]),'Loaded residual identity differs')
   require(canonical(engine.spec)==canonical(self.specs[lang]['75']) and canonical(engine.v9spec)==canonical(self.specs[lang]['9']),'Loaded language/route specification differs')
   require(engine.spec_identity==digest(engine.spec) and engine.v9_spec_identity==digest(engine.v9spec),'Loaded specification state differs')
   require(Path(engine.alignment.__file__).resolve()==Path(self.closure._registration['packages']['roguard']['root']).resolve()/'role_alignment.py','Loaded alignment origin differs')
 def _component_identity(self):
  return {lang:(id(component),id(getattr(component.predict_detailed,'__func__',component.predict_detailed))) for lang,component in self.components.items()}
 def _guard(self,*,full=False):
  require(set(self.components)=={self.active_language},'Active language/component binding changed')
  (self.closure.verify if full else self.closure.cheap_verify)();self._verify_installed_components();require(self._component_identity()==self._component_bindings,'Detailed component binding changed');require(sha(__file__)==self.contract['wrapper_source_sha256'],'Wrapper source changed');require(digest(self.contract)==self.identity and digest(self.specs)==self.spec_state and digest(self.manifest_values)==self.manifest_state and digest(self.selector.spec)==self.selector_state,'Bound wrapper state mutated')
 def _detail(self,inference):
  detail=self.components[inference['language']].predict_detailed(deepcopy(inference),focus=len(inference['turns'])-1)
  require(set(getattr(detail,'_fields',()))=={'response','logits','route','spec_sha256'},'Detailed record contract differs')
  route=detail.route;require(route in ('residual','original_v9','original_v9_unsupported'),'Unknown route');lang=inference['language'];which='75' if route=='residual' else '9';spec=self.specs[lang][which]
  require(detail.spec_sha256==(SPEC75 if which=='75' else SPEC9)[lang],'Detailed route spec differs');raw=deepcopy(detail.response)
  require(raw['language']==lang and type(raw['focus']) is int and raw['focus']==len(inference['turns'])-1 and raw['input_digest']==input_identity(inference),'Native prefix identity differs')
  require(raw['model_identity']==(self.manifest_ids[lang] if route=='residual' else SPEC9[lang]),'Native model identity differs')
  if route=='original_v9_unsupported':require(detail.logits is None and raw['score'] is None and raw['predicted'] is None,'Unsupported response cannot contain confidence');return raw,None,route,None,None
  z=detail.logits;require(type(z) is tuple and len(z)==19 and all(type(v) is float and math.isfinite(v) for v in z),'Finite immutable19float logits required')
  # Exact float32 representability, including signed-zero, without a NumPy dependency.
  import struct
  require(all(struct.unpack('<f',struct.pack('<f',v))[0].hex()==v.hex() for v in z),'Detailed logits changed float32 values')
  score=float(_score(z,spec));require(type(raw['score']) is float and raw['score'].hex()==score.hex() and type(raw['predicted']) is int and raw['predicted']==int(score>=spec['threshold']),'Native score/decision differs')
  probabilities=coupled_worlds(z,spec) if route=='residual' else factors_from_logits(z);mass=math.fsum(v for v,world in zip(probabilities,WORLDS) if _complete_action(world)!='CONTINUE')
  equal=route=='residual' or lang=='ro'
  if equal:require(abs(mass-score)<=MASS_ATOL,'Semantic policy mass differs')
  return raw,probabilities,route,dict(semantic_review_mass=mass,score_mass_equivalence_required=equal,semantic_vs_binary_difference=abs(mass-score)),z
 def batch(self):return VerifiedBatch(self)
 def predict(self,payload,*,focus=None,calibration=None):
  self._require_active(payload)
  with self.batch() as batch:result=batch.predict(payload,focus=focus,calibration=calibration)
  return result
 def predict_many(self,requests):
  require(type(requests) is list and requests,'Nonempty explicit request list required')
  for request in requests:
   require(type(request) is dict and 'payload' in request and set(request)<={'payload','focus','calibration'},'Request schema differs');self._require_active(request['payload'])
  with self.batch() as batch:
   result=[]
   for request in requests:
    require(type(request) is dict and 'payload' in request and set(request)<={'payload','focus','calibration'},'Request schema differs')
    result.append(batch.predict(**request))
  return result
 def _predict(self,payload,*,focus=None,calibration=None):
  self._require_active(payload);self._guard();validate_conversation(payload);focus=len(payload['turns'])-1 if focus is None else focus;original=deepcopy(prefix(payload,focus));require(original['turns'][-1]['role']=='minor','Focal minor required')
  direct=None;raw=None;probabilities=None;route=None;mass=None;field_logits=None;view=None;inference=original;segments=[];unsupported_reason=None
  if self.mode=='direct-first' and len(original['turns'])<=12:
   direct=self._detail(original);raw,probabilities,route,mass,field_logits=direct
  need_memory=self.mode=='uniform' or len(original['turns'])>12 or (direct is not None and route=='original_v9_unsupported')
  if need_memory:
   view,inference=prepare_memory(original,self.selector,self.budget,preserve_complete=True);verify_view(original,view)
   if view['mandatory_overflow'] or not inference['turns'] or len(inference['turns'])>12:
    unsupported_reason='MANDATORY_MEMORY_OVERFLOW' if view['mandatory_overflow'] else 'MEMORY_VIEW_EXCEEDS_ORIGINAL_TURN_CONTRACT';raw=None;probabilities=None;route='memory_unsupported';mass=None;field_logits=None
   else:
    segments=rendering_segments(original,view,inference);raw,probabilities,route,mass,field_logits=self._detail(inference)
  evidence=[{k:s[k] for k in ('turn','start','end')} for s in (view['spans'] if view is not None else units(original))];check_evidence(original,evidence)
  score=None if raw is None else raw['score'];predicted=None if raw is None else raw['predicted'];supported=predicted is not None
  if not supported:
   probabilities=[1/432]*432;sets=dict(hypotheses=deepcopy(WORLDS),reason='UNSUPPORTED_OR_EMPTY_SET',calibration_digest=None,coverage_assumptions_verified=False,outside_ontology_ruled_out=False)
  else:sets=None
  if calibration is not None:
   # Uses an already-frozen artifact only. This method never fits calibration.
   sets=predict_set(probabilities,calibration,language=original['language'],pipeline_digest=self.identity,supported=supported)
  result=dict(format='roguard-full-route-structured-prediction-research-1',language=original['language'],focus=focus,input_digest=input_identity(original),inference_input_digest=None if not inference['turns'] else input_identity(inference),model_identity=self.identity,pipeline_digest=self.identity,predicted=predicted,score=score,native_response=raw,field_logits=field_logits,direct_unsupported_response=None if direct is None or direct[2]!='original_v9_unsupported' else direct[0],route=route,semantic_score_relation=mass,probabilities=probabilities,probabilities_source='UNSUPPORTED_UNIFORM_PLACEHOLDER' if not supported else 'SAME_FORWARD_ROUTE_LOGITS',evidence=evidence,evidence_scope='retained_source_units_not_verified_semantic_support',used_memory=view is not None,memory=view,inference_to_original_segments=segments,unsupported_reason=unsupported_reason,qualified=False,establishes_safety=False,permits_external_action=False)
  if sets is not None:result['interpretation_set']=sets
  self._guard();return result

class VerifiedBatch:
 """Boundary-verified lifecycle. Streamed results remain pending until close."""
 def __init__(self,wrapper):self.wrapper=wrapper;self.results=[];self.closed=False;self.entered=False;self.failure=None
 def __enter__(self):
  require(not self.entered and self.wrapper._active_batch is None,'Batch already active or used');self.wrapper._guard(full=True);self.entered=True;self.wrapper._active_batch=self;return self
 def predict(self,payload,*,focus=None,calibration=None):
  require(self.entered and not self.closed and self.failure is None and self.wrapper._active_batch is self,'Prediction requires active successful batch')
  try:result=self.wrapper._predict(payload,focus=focus,calibration=calibration)
  except BaseException as exc:self.failure=exc;raise
  result['closure_verification']=dict(state='PENDING_BATCH_CLOSE',scope='full_disk_closure_at_batch_boundaries;in_memory_and_import_origins_each_row');self.results.append(result);return result
 def __exit__(self,typ,error,tb):
  closing=None
  try:self.wrapper._guard(full=True)
  except BaseException as exc:closing=exc
  finally:
   self.wrapper._active_batch=None;self.closed=True
   for result in self.results:result['closure_verification']['state']='FAILED_BATCH' if error is not None or closing is not None or self.failure is not None else 'PASS'
  if closing is not None:
   if error is not None:raise BaseExceptionGroup('Prediction and closing verification failed',[error,closing])
   raise closing
  if error is None and self.failure is not None:raise ValueError('Batch contains a failed prediction') from self.failure
  return False
