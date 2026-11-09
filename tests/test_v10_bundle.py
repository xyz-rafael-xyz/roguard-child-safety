import importlib.util,json,hashlib,types,threading,weakref,gc,copy
from pathlib import Path
import pytest
import sys
import roguard.v10_bundle as m

def bind(path,root):return dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=m._sha(path))
def write(path,value):path.write_text(m._canonical(value));return path

def calibration(pipeline):
 from roguard.assurance import digest,policy_digest
 q=dict(format='roguard-joint-calibration-v10-research1',pipeline_digest=pipeline,policy_digest=policy_digest(),quantiles={'ro':.5,'uk':.5});q['digest']=digest(q);return q

def fixture(root,cal=None):
 root.mkdir(exist_ok=True);a=write(root/'structured-inventory.json',{'kind':'structured','synthetic':True});b=write(root/'response-inventory.json',{'kind':'response','synthetic':True})
 p=dict(format=m.FORMAT,qualified=False,languages=['ro','uk'],inventories={'structured':bind(a,root),'response':bind(b,root)},structured=dict(mode='direct-first',budget=1200,pipeline_digest='1'*64),response=dict(reference_identity='2'*64,parent_identity='3'*64,artifact_identity='4'*64),calibration=None)
 if cal is not None:p['calibration']=bind(write(root/'calibration.json',cal),root)
 path=write(root/'bundle.json',p);return path,p

def payload(lang='ro',text='Salut.'):return dict(language=lang,turns=[dict(role='minor',text=text)])

class Closure:
 def __init__(self):self.bad=False;self.calls=0
 def verify(self):
  self.calls+=1
  if self.bad:raise ValueError('closing fixture mutation')

class Factory:
 def __init__(self):self.events=[];self.live=0;self.backends=[];self.fail_call=False;self.fail_close=False;self.fail_activate=False;self.callback=None
 def prepare(self,*args):self.events.append(('prepare',));return {'structured':Closure(),'response':Closure()}
 def activate(self,task,language,spec,closures):
  assert self.live==0,'overlapping native lifetimes';self.events.append(('activate',task,language))
  if self.fail_activate:raise RuntimeError('activation failed')
  self.live+=1;owner=self
  class Backend:
   closed=False
   def predict_many(self,requests):
    if owner.callback:owner.callback()
    if owner.fail_call:raise RuntimeError('inference failed')
    return [dict(language=r['payload']['language'],text=r['payload']['turns'][-1]['text'],qualified=False,calibration=r['calibration'],focus=r['focus']) for r in requests]
   def review_response(self,draft,lang,context):
    if owner.fail_call:raise RuntimeError('inference failed')
    return dict(language=lang,draft=draft,context=context,qualified=False,permits_external_action=False)
   def close(self):
    if self.closed:return
    self.closed=True;owner.live-=1;owner.events.append(('close',task,language))
    if owner.fail_close:raise RuntimeError('cleanup failed')
  out=Backend();self.backends.append(weakref.ref(out));return out

@pytest.fixture(autouse=True)
def reset_lease(monkeypatch):
 monkeypatch.setattr(m,'_OWNER',None);monkeypatch.setattr(m,'_POISONED',False)
 yield
 m._OWNER=None;m._POISONED=False

def model(root,cal=None):
 path,p=fixture(root,cal);f=Factory();return m.V10Bundle(path,m._sha(path),_factory=f),f,path,p

def test_lazy_batch_language_order_and_task_cleanup(tmp_path):
 obj,f,path,p=model(tmp_path);assert f.events==[('prepare',)] and f.live==0
 out=obj.predict_many([payload('uk','one'),payload('ro','two'),payload('uk','three')]);assert [r['text'] for r in out]==['one','two','three']
 assert f.events==[('prepare',),('activate','structured','ro'),('close','structured','ro'),('activate','structured','uk')]
 response=obj.review_response('Draft','ro',context='Declared context');assert response['context']=='Declared context' and not response['qualified']
 assert f.events[-2:]==[('close','structured','uk'),('activate','response','ro')]
 obj.close();obj.close();assert f.live==0 and all(r() is None for r in f.backends)
 with pytest.raises(ValueError,match='closed'):obj.predict(payload())

def test_two_instances_lease_does_not_poison_valid_owner(tmp_path):
 a,fa,*_=model(tmp_path/'a');b,fb,*_=model(tmp_path/'b');a.predict(payload())
 with pytest.raises(ValueError,match='other active'):b.predict(payload())
 assert not m._POISONED and fb.live==0;a.predict(payload());a.close();b.predict(payload());b.close()

@pytest.mark.parametrize('failure',['inference','activation','cleanup'])
@pytest.mark.skipif(sys.version_info < (3, 12), reason='Native exception cleanup contract requires Python 3.12+')
def test_failures_cleanup_and_poison(tmp_path,failure):
 obj,f,*_=model(tmp_path)
 if failure=='activation':f.fail_activate=True
 else:
  obj.predict(payload());setattr(f,'fail_call' if failure=='inference' else 'fail_close',True)
 with pytest.raises((RuntimeError,BaseExceptionGroup)):
  if failure=='cleanup':obj.review_response('draft','ro',context='')
  else:obj.predict(payload())
 assert f.live==0 and m._POISONED
 with pytest.raises(ValueError,match='restart'):obj.predict(payload())
 obj.close()

def test_mixed_batch_rejects_all_invalid_before_activation(tmp_path):
 obj,f,*_=model(tmp_path)
 for pp in [[payload(),payload('fr')],[payload(),dict(language='ro',turns=[dict(role='assistant',text='x')])]]:
  with pytest.raises(ValueError):obj.predict_many(pp)
 assert f.live==0 and f.events==[('prepare',)];obj.close()

def test_context_and_unsupported_plan_are_explicit(tmp_path):
 obj,f,*_=model(tmp_path)
 with pytest.raises(TypeError):obj.review_response('draft','ro')
 with pytest.raises(ValueError):obj.review_response('draft','ro',context=None)
 with pytest.raises(TypeError):obj.predict(payload(),plan=True)
 assert f.events==[('prepare',)] and not obj.profile['supports_plan'];obj.close()

def test_calibration_override_default_and_inference_identity(tmp_path):
 cal=calibration('1'*64);obj,f,path,p=model(tmp_path/'with',cal);bare,fb,*_=model(tmp_path/'without')
 assert obj.inference_identity==bare.inference_identity and obj.bundle_sha256!=bare.bundle_sha256
 assert obj.predict(payload())['calibration']==cal
 override=calibration('1'*64);override['quantiles']['ro']=.8;from roguard.assurance import digest
 override['digest']=digest({k:v for k,v in override.items() if k!='digest'})
 assert obj.predict(payload(),calibration=override)['calibration']==override
 bad=calibration('a'*64)
 with pytest.raises(ValueError,match='another pipeline'):obj.predict(payload(),calibration=bad)
 obj.close();bare.close()

@pytest.mark.parametrize('mutation',['absolute','traversal','backslash','symlink','bytes_bool','hash','extra','qualified','budget_bool'])
def test_manifest_errors_before_factory(tmp_path,mutation):
 path,p=fixture(tmp_path);b=p['inventories']['structured']
 if mutation=='absolute':b['path']=str(tmp_path/'structured-inventory.json')
 if mutation=='traversal':b['path']='../outside.json'
 if mutation=='backslash':b['path']='sub\\file.json'
 if mutation=='symlink':(tmp_path/'alias.json').symlink_to(tmp_path/'structured-inventory.json');b['path']='alias.json'
 if mutation=='bytes_bool':b['bytes']=True
 if mutation=='hash':b['sha256']='0'*64
 if mutation=='extra':p['download_url']='https://invalid'
 if mutation=='qualified':p['qualified']=True
 if mutation=='budget_bool':p['structured']['budget']=True
 write(path,p);f=Factory()
 with pytest.raises(ValueError):m.V10Bundle(path,m._sha(path),_factory=f)
 assert not f.events

def test_bound_inventory_mutation_rejects_before_new_encoder(tmp_path):
 obj,f,path,p=model(tmp_path);(tmp_path/'structured-inventory.json').write_text('altered')
 with pytest.raises(ValueError):obj.predict(payload())
 assert f.live==0;obj.close()

def test_native_lock_serializes_call_and_close(tmp_path):
 obj,f,*_=model(tmp_path);entered=threading.Event();release=threading.Event();closed=threading.Event();errors=[]
 def callback():entered.set();assert release.wait(2)
 f.callback=callback
 def call():
  try:obj.predict(payload())
  except BaseException as e:errors.append(e)
 a=threading.Thread(target=call);a.start();assert entered.wait(2)
 b=threading.Thread(target=lambda:(obj.close(),closed.set()));b.start();assert not closed.wait(.05);release.set();a.join(2);b.join(2)
 assert closed.is_set() and not errors and f.live==0

@pytest.mark.skipif(sys.version_info < (3, 12), reason='Native exception cleanup contract requires Python 3.12+')
def test_real_backend_detaches_nested_refs_even_closure_error():
 closure=Closure();backend=m._Backend(closure);holder=types.SimpleNamespace();engine=types.SimpleNamespace(model=holder,tokenizer=object(),make_cache=object(),mx=types.SimpleNamespace(clear_cache=lambda:None));optional=types.SimpleNamespace(parent=engine,original_head=object(),residual=object(),alignment=object());component=types.SimpleNamespace(_parent=optional);wrapper=types.SimpleNamespace(components={'ro':component})
 backend.wrapper=wrapper;backend.component=component;backend.optional=optional;backend.engine=engine;closure.bad=True
 with pytest.raises(BaseExceptionGroup):backend.close()
 assert not wrapper.components and component._parent is None and optional.parent is None and optional.original_head is None and engine.model is None and engine.tokenizer is None and backend.engine is None

def test_public_loader_has_no_factory_injection(tmp_path,monkeypatch):
 path,_=fixture(tmp_path);f=Factory();monkeypatch.setattr(m,'_ActualFactory',lambda:f)
 obj=m.load_v10(path,expected_sha256=m._sha(path));assert f.events==[('prepare',)];obj.close()
 with pytest.raises(TypeError):m.load_v10(path,expected_sha256=m._sha(path),factory=f)

def test_abandoned_owner_remains_strongly_leased(tmp_path):
 obj,f,*_=model(tmp_path/'a');obj.predict(payload());ref=weakref.ref(obj);del obj;gc.collect()
 assert ref() is not None and m._OWNER is ref()
 other,fo,*_=model(tmp_path/'b')
 with pytest.raises(ValueError,match='other active'):other.predict(payload())
 assert fo.live==0;m._OWNER.close();other.predict(payload());other.close();assert ref() is None

@pytest.mark.parametrize('name',['C:drive.json','decomposed-e\u0301.json'])
def test_nonportable_paths_rejected(name,tmp_path):
 path,p=fixture(tmp_path);p['inventories']['structured']['path']=name;write(path,p);f=Factory()
 with pytest.raises(ValueError,match='Portable'):m.V10Bundle(path,m._sha(path),_factory=f)
 assert not f.events

def test_duplicate_bundle_and_calibration_json_rejected(tmp_path):
 path,p=fixture(tmp_path);raw=m._canonical(p);path.write_text(raw[:-1]+',"qualified":false}');f=Factory()
 with pytest.raises(ValueError,match='Duplicate'):m.V10Bundle(path,m._sha(path),_factory=f)
 cal=calibration('1'*64);cp=tmp_path/'calibration.json';cp.write_text(m._canonical(cal)[:-1]+',"pipeline_digest":"'+('1'*64)+'"}');p['calibration']=bind(cp,tmp_path);write(path,p)
 with pytest.raises(ValueError,match='Duplicate'):m.V10Bundle(path,m._sha(path),_factory=f)
 assert not f.events

def test_actual_factory_constructor_wiring_with_stubbed_native_classes(tmp_path,monkeypatch):
 from roguard import experimental_role_residual as residual,experimental_structured as structured,trajectory_model,gemma4_contract,gemma4_review,gemma4_response_attention
 from roguard.response_offset import runtime as offset,closure as response_closure
 calls=[]
 class Assets(Closure):
  def asset_path(self,name):return tmp_path/'base'/name[5:] if name.startswith('base:') else tmp_path/name
  def asset_sha(self,name):return hashlib.sha256(name.encode()).hexdigest()
 a=Assets();closures={'structured':a,'response':a};_,p=fixture(tmp_path)
 monkeypatch.setattr(m.shutil,'disk_usage',lambda _:types.SimpleNamespace(free=100*1024**3))
 def load_residual(**kwargs):
  calls.append(('residual',kwargs));engine=types.SimpleNamespace(mx=types.SimpleNamespace(clear_cache=lambda:None),model=object(),tokenizer=object(),make_cache=object());return types.SimpleNamespace(_parent=types.SimpleNamespace(parent=engine,original_head=object(),residual=object(),alignment=object()))
 monkeypatch.setattr(residual,'load_experimental_role_residual',load_residual)
 monkeypatch.setattr(trajectory_model,'TrajectoryModel',lambda path,**kw:calls.append(('selector',path,kw)))
 def wrap(components,selector,closure,**kw):calls.append(('structured',components.copy(),kw));return types.SimpleNamespace(identity=p['structured']['pipeline_digest'],components=components)
 monkeypatch.setattr(structured,'StructuredWrapper',wrap)
 f=m._ActualFactory();b=f.activate('structured','uk',p,closures)
 assert calls[0][1]['parent75_spec']==tmp_path/'uk_spec75' and calls[0][1]['original_v9_adapter']==tmp_path/'uk_adapter9' and calls[0][1]['enabled'] is True
 assert calls[-1][2]==dict(active_language='uk',mode='direct-first',budget=1200);b.close()
 def reference(spec,weights,base,**kwargs):calls.append(('reference',spec,weights,base,kwargs));return types.SimpleNamespace(identity=p['response']['reference_identity'],_engine=None)
 def parent(spec,weights,ref,**kwargs):calls.append(('parent',spec,weights,ref,kwargs));return types.SimpleNamespace(identity=p['response']['parent_identity'],reference=ref,_readout=None)
 def fitted(parent,closure,**kwargs):calls.append(('offset',parent,closure,kwargs));return types.SimpleNamespace(artifact_identity=p['response']['artifact_identity'],parent=parent)
 monkeypatch.setattr(gemma4_review,'Gemma4ReviewModel',reference);monkeypatch.setattr(gemma4_response_attention,'Gemma4ResponseAttentionModel',parent);monkeypatch.setattr(offset,'Frozen59TextOffset',fitted)
 b=f.activate('response','ro',p,closures)
 assert calls[-3][1:3]==(tmp_path/'reference_spec',tmp_path/'reference_weights') and calls[-2][1:3]==(tmp_path/'response59_spec',tmp_path/'response59_weights')
 assert calls[-1][3]['closure_type'] is response_closure.Closure;b.close()

def test_actual_prepare_checks_both_receipts(tmp_path,monkeypatch):
 import sys
 from roguard import experimental_structured
 from roguard.response_offset import closure
 path,p=fixture(tmp_path);out=tmp_path/'local';out.mkdir();calls=[]
 def rebase(path,**kwargs):
  kind='structured' if 'structured' in path.name else 'response';calls.append((kind,path,kwargs));return types.SimpleNamespace(path=out/(kind+'.json'),sha256='a'*64,receipt_sha256='b'*64,kind=kind)
 fake=types.ModuleType('roguard.v10_closure');fake.rebase_closure=rebase;monkeypatch.setitem(sys.modules,'roguard.v10_closure',fake)
 monkeypatch.setattr(experimental_structured,'Closure',lambda *a:types.SimpleNamespace(identity='b'*64));monkeypatch.setattr(closure,'Closure',lambda *a:types.SimpleNamespace(identity='b'*64))
 result=m._ActualFactory().prepare(tmp_path,p,out);assert set(result)=={'structured','response'} and [x[0] for x in calls]==['structured','response']
 monkeypatch.setattr(closure,'Closure',lambda *a:types.SimpleNamespace(identity='c'*64))
 with pytest.raises(ValueError,match='receipt'):m._ActualFactory().prepare(tmp_path,p,out)

@pytest.mark.parametrize('change',['valid_replacement','remove'])
def test_bound_default_calibration_cannot_silently_change(tmp_path,change):
 from roguard.assurance import digest
 obj,f,*_=model(tmp_path,calibration('1'*64))
 if change=='remove':obj._default_calibration=None
 else:
  obj._default_calibration['quantiles']['ro']=.9;obj._default_calibration['digest']=digest({k:v for k,v in obj._default_calibration.items() if k!='digest'})
 with pytest.raises(ValueError,match='Bound default calibration'):obj.predict(payload())
 assert f.events==[('prepare',)] and f.live==0;obj.close()
