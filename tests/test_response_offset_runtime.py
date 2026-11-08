import contextlib,hashlib,importlib.util,json,sys,types,unicodedata
from pathlib import Path
import pytest

if sys.version_info < (3, 12):
 pytest.skip("Experimental neural runtime requires Python >=3.12", allow_module_level=True)
np = pytest.importorskip("numpy")
from roguard.response_offset import runtime as rt, closure as cl, lexical as lex
HERE=Path(rt.__file__).resolve().parent
from roguard import gemma4_response_attention as native

def make_model(tmp_path,monkeypatch,zero=False):
 package=tmp_path/'package';package.mkdir();(package/'fixture.py').write_text('marker=1\n');assetdir=tmp_path/'assets';assetdir.mkdir()
 weight=assetdir/'parent.bin';weight.write_bytes(b'synthetic-native-parent-no-tensors');runtime_spec=assetdir/'runtime.json';runtime_spec.write_text('{}');training=assetdir/'training.json';training.write_text('{"training":true}')
 monkeypatch.setattr(rt,'PARENT_WEIGHTS',cl.sha(weight));monkeypatch.setattr(rt,'RUNTIME_SPEC',cl.sha(runtime_spec));monkeypatch.setattr(rt,'TRAINING_SPEC',cl.sha(training))
 lexical=lex.fit_lexicon(['label0 suitable words','label1 alternate words']*2);lexical['train_rows']=13076
 vocabfile=assetdir/'lexicon.json';vocabfile.write_text(rt.canonical(lexical));width=sum(len(lexical['channels'][x]['vocabulary']) for x in ('word','char'));theta=np.zeros(5*(width+1),np.float64)
 if not zero:theta[-4]=3. # synthetic bias changes the public response class
 thetafile=assetdir/'theta.npy';np.save(thetafile,theta)
 # Real public method and real72-array restoration, with tiny fake features/readout only.
 originals={f'factor{i}':np.asarray([[float(i)]],np.float32) for i in range(72)};holders={key:types.SimpleNamespace(value=value) for key,value in originals.items()}
 slots={key:(obj,'value') for key,obj in holders.items()};monkeypatch.setattr(native,'_adapter_slots',lambda base:slots)
 state={'calls':0,'clear':0,'factory':0,'responses':[],'unsupported':False,'failure':None,'raw':[-0.,0.,-1.,-2.,-3.]}
 mx=types.SimpleNamespace(clear_cache=lambda:state.__setitem__('clear',state['clear']+1));engine=types.SimpleNamespace(model=types.SimpleNamespace(base=object()),mx=mx)
 class Readout:
  def __init__(self,*args):self.adapters={key:np.asarray([[99.]],np.float32) for key in originals};state['factory']+=1
  def logits(self,features):return list(state['raw'])
 readout=Readout();monkeypatch.setattr(native,'_AdapterReadout',Readout)
 def features(engine,payload):
  state['calls']+=1;state['payload']=payload
  if state['failure'] is not None:raise state['failure']
  return None if state['unsupported'] else np.zeros((1,2560),np.float32)
 monkeypatch.setattr(native,'_features',features);monkeypatch.setattr(native,'_Engine',lambda *args:engine)
 ref=types.SimpleNamespace(identity='synthetic-reference',_args=(tmp_path,weight,cl.sha(weight)),_spec={'synthetic':True},_engine=engine)
 parent=object.__new__(native.Gemma4ResponseAttentionModel);parent._spec={'adapter_sha256':rt.PARENT_WEIGHTS};parent.reference=ref;parent._weights=weight;parent._readout=readout;parent._reference_binding=(ref.identity,ref._args,native.digest(ref._spec));parent.identity=native.digest(dict(format=native.FORMAT,artifact=rt.RUNTIME_SPEC,reference=ref.identity))
 original_method=native.Gemma4ResponseAttentionModel.review_response
 def public(draft,language,context=''):
  response=original_method(parent,draft,language,context);state['responses'].append(response);return response
 parent.review_response=public
 assets={'runtime_source':HERE/'runtime.py','closure_source':HERE/'closure.py','lexical_source':HERE/'lexical.py','response59_spec':runtime_spec,'response59_training_manifest':training,'response59_weights':weight,'offset_theta':thetafile,'offset_lexicon':vocabfile}
 manifest=dict(format=rt.FORMAT,qualified=False,labels=list(rt.LABELS),parent_identity=parent.identity,reference_identity=ref.identity,parent59_runtime_spec_sha256=rt.RUNTIME_SPEC,parent59_training_spec_sha256=rt.TRAINING_SPEC,parent59_adapter_sha256=rt.PARENT_WEIGHTS,lexical_source_sha256=rt.LEXICAL_SOURCE,runtime_source_sha256=cl.sha(HERE/'runtime.py'),closure_source_sha256=cl.sha(HERE/'closure.py'),theta_sha256=cl.sha(thetafile),lexicon_sha256=cl.sha(vocabfile),selected_arm='new_data',selected_candidate='zero' if zero else 'lambda0.001',query_sha256=native.QUERY_SHA256,probabilities_calibrated=False,**{key:'c'*64 for key in rt.PROVENANCE})
 mf=assetdir/'manifest.json';mf.write_text(rt.canonical(manifest));assets['offset_manifest']=mf
 reg=dict(format='roguard-response-offset-installed-closure-1',scope='synthetic_fixture',qualified=False,packages={'synthetic_offset_fixture_pkg':{'root':str(package),'files':cl.python_inventory(package)}},assets={k:{'path':str(p),**cl.file_record(p)} for k,p in assets.items()},required_asset_ids=sorted(assets),native_dependencies={},asset_edges=[{'from':'offset_manifest','keys':[key],'to':target} for key,target in [('theta_sha256','offset_theta'),('lexicon_sha256','offset_lexicon'),('runtime_source_sha256','runtime_source'),('closure_source_sha256','closure_source'),('lexical_source_sha256','lexical_source'),('parent59_runtime_spec_sha256','response59_spec'),('parent59_training_spec_sha256','response59_training_manifest'),('parent59_adapter_sha256','response59_weights')]],declared_supported_runtime_contract={'scope':'synthetic_no_encoder'})
 regfile=tmp_path/'closure.json';regfile.write_text(rt.canonical(reg));closure=cl.Closure(regfile,cl.sha(regfile));wrapper=rt.Frozen59TextOffset(parent,closure,closure_type=cl.Closure,native=native)
 return wrapper,parent,state,holders,originals,assets

def restored(parent,readout,holders,originals):
 assert parent._readout is readout
 assert all(holders[k].value is v and holders[k].value.tobytes()==v.tobytes() for k,v in originals.items())

def test_actual_public_method_corrected_flags_and_same_call(tmp_path,monkeypatch):
 model,parent,state,holders,originals,assets=make_model(tmp_path,monkeypatch);readout=parent._readout
 d=model.review_response_detailed('label0 suitable words','ro','Context e\u0301')
 assert state['calls']==1 and d.parent_logits==tuple(state['raw']) and d.logits[1]==3.
 assert d.response['label']=='boundary' and d.response['review_requested'] is True
 assert d.response['probabilities']==dict(zip(native.LABELS,native._probabilities(list(d.logits))))
 assert d.response['model_identity']==model.artifact_identity and d.parent_identity==parent.identity and d.route=='text_offset'
 assert d.input_digest==native.digest(state['payload']) and state['payload']['turns'][0]['text']==unicodedata.normalize('NFC','Context e\u0301')
 assert isinstance(d.parent_logits,tuple) and isinstance(d.logits,tuple);restored(parent,readout,holders,originals)
 assert np.asarray(d.parent_logits,np.float32)[0].tobytes()==np.asarray(-0.,np.float32).tobytes()

def test_zero_exact_parent_object_identity_and_detail(tmp_path,monkeypatch):
 model,parent,state,holders,originals,_=make_model(tmp_path,monkeypatch,True);readout=parent._readout
 response=model.review_response('label0 suitable words','uk');assert response is state['responses'][-1] and response['model_identity']==parent.identity and model.identity==parent.identity
 detail=model.review_response_detailed('label0 suitable words','uk');assert detail.response is state['responses'][-1] and detail.logits is detail.parent_logits and detail.route=='zero_parent'
 assert state['calls']==2;restored(parent,readout,holders,originals)

@pytest.mark.parametrize('zero',[False,True])
def test_unsupported_and_failure_restore(tmp_path,monkeypatch,zero):
 model,parent,state,holders,originals,_=make_model(tmp_path,monkeypatch,zero);readout=parent._readout;state['unsupported']=True
 result=model.review_response_detailed('label0 suitable words','ro');assert result.route=='unsupported_parent' and result.parent_logits is None and result.logits is None and result.response['abstained'] is True and result.response['label'] is None
 restored(parent,readout,holders,originals);state['unsupported']=False;state['failure']=RuntimeError('synthetic feature failure')
 with pytest.raises(RuntimeError,match='synthetic feature failure'):model.review_response_detailed('label0 suitable words','ro')
 restored(parent,readout,holders,originals);assert state['clear']==2

def test_no_second_encoder_and_input_validation_before_lazy_init(tmp_path,monkeypatch):
 model,parent,state,holders,originals,_=make_model(tmp_path,monkeypatch);parent.reference._engine=None;parent._readout=None;model._readout_identity=None
 engines=[];engine=types.SimpleNamespace(model=types.SimpleNamespace(base=object()),mx=types.SimpleNamespace(clear_cache=lambda:None));monkeypatch.setattr(native,'_Engine',lambda *args:(engines.append('one') or engine))
 with pytest.raises(ValueError):model.review_response_detailed(' ','ro')
 assert not engines and state['calls']==0
 model.review_response_detailed('label0 suitable words','ro');model.review_response_detailed('label0 suitable words','uk')
 assert engines==['one'] and state['calls']==2 and state['factory']==2 # one initial fixture +one actual lazy readout


def test_batch_closing_mutation_and_exception_group(tmp_path,monkeypatch):
 model,parent,state,holders,originals,assets=make_model(tmp_path,monkeypatch);readout=parent._readout
 with pytest.raises(BaseExceptionGroup) as result:
  with model.batch():
   assets['offset_theta'].write_bytes(b'corrupt');state['failure']=RuntimeError('first cause');model.review_response_detailed('label0 suitable words','ro')
 assert any(isinstance(x,RuntimeError) and str(x)=='first cause' for x in result.value.exceptions)
 assert any(isinstance(x,ValueError) for x in result.value.exceptions);restored(parent,readout,holders,originals)

def test_loaded_parameter_mutation_rejected_at_batch_close(tmp_path,monkeypatch):
 model,*_=make_model(tmp_path,monkeypatch)
 with pytest.raises(ValueError,match='Loaded lexical parameters changed'):
  with model.batch():model._lexicon['train_rows']=13077


def test_portable_float64_math_against_csr(tmp_path,monkeypatch):
 pytest.importorskip("scipy")
 model,*_=make_model(tmp_path,monkeypatch);rng=np.random.default_rng(5101);theta=rng.normal(0,.03,size=model._theta.shape).astype(np.float64);texts=['label0 suitable words','label1 alternate words','Не надсилай пароль.','Nu cer date private.']*16
 x=lex.matrix([None]*len(texts),texts,model._lexicon,'text_only');width=x.shape[1];parents=rng.normal(size=(len(texts),5)).astype(np.float32);reference=parents.astype(np.float64)+x@theta[:5*width].reshape(5,width).T+theta[5*width:]
 actual=np.asarray([rt.adjusted_logits(tuple(map(float,p)),draft,model._lexicon,theta,model._lexical) for p,draft in zip(parents,texts)])
 assert np.max(abs(actual-reference))<=1e-10 and np.array_equal(actual.argmax(1),reference.argmax(1))

def test_pure_runtime_has_no_scipy_dependency(tmp_path,monkeypatch):
 model,*_=make_model(tmp_path,monkeypatch);monkeypatch.setitem(sys.modules,'scipy',None)
 result=rt.adjusted_logits((1.,0.,-1.,-2.,-3.),'label0 suitable words',model._lexicon,model._theta,model._lexical);assert len(result)==5

@pytest.mark.parametrize('bad',[[],[0.]*4,[float('nan')]*5,[1e100]*5,[1,0.,0.,0.,0.]])
def test_invalid_raw_five_reject(tmp_path,monkeypatch,bad):
 model,*_=make_model(tmp_path,monkeypatch)
 with np.errstate(over='ignore',invalid='ignore'):
  with pytest.raises((ValueError,OverflowError)):rt.adjusted_logits(bad,'label0 suitable words',model._lexicon,model._theta,model._lexical)

@pytest.mark.parametrize('zero',[False,True])
def test_bound_readout_swap_rejected_before_forward(tmp_path,monkeypatch,zero):
 model,parent,state,*_=make_model(tmp_path,monkeypatch,zero)
 parent._readout=types.SimpleNamespace(adapters={},logits=lambda features:[0.,9.,0.,0.,0.])
 with pytest.raises(ValueError,match='Parent readout replaced'):model.review_response('label0 suitable words','ro')
 assert state['calls']==0


def test_foreign_thread_cannot_borrow_unclosed_batch(tmp_path,monkeypatch):
 import threading
 model,parent,state,holders,originals,assets=make_model(tmp_path,monkeypatch);opened=threading.Event();attempt=threading.Event();release=threading.Event();returned=threading.Event();errors=[];responses=[]
 def first():
  try:
   with model.batch():
    assets['offset_theta'].write_bytes(b'corrupt');opened.set();assert release.wait(3)
  except BaseException as exc:errors.append(('first',exc))
 def second():
  assert opened.wait(3);attempt.set()
  try:responses.append(model.review_response('label0 suitable words','ro'))
  except BaseException as exc:errors.append(('second',exc))
  finally:returned.set()
 x=threading.Thread(target=first);y=threading.Thread(target=second);x.start();y.start();assert attempt.wait(3)
 assert not returned.wait(.1);release.set();x.join(3);y.join(3)
 assert not x.is_alive() and not y.is_alive() and not responses and {who for who,_ in errors}=={'first','second'} and all(isinstance(exc,ValueError) for _,exc in errors)
 assert state['calls']==0

@pytest.mark.parametrize('failed_first_call',[False,True])
def test_first_zero_default_binds_lazy_readout(tmp_path,monkeypatch,failed_first_call):
 model,parent,state,*_=make_model(tmp_path,monkeypatch,True);parent._readout=None;model._readout_identity=None
 if failed_first_call:
  state['failure']=RuntimeError('first zero call fails after readout initialization')
  with pytest.raises(RuntimeError):model.review_response('label0 suitable words','ro')
 else:
  response=model.review_response('label0 suitable words','ro');assert response is state['responses'][-1]
 assert parent._readout is not None and model._readout_identity==id(parent._readout)
 previous_calls=state['calls'];previous_factories=state['factory'];parent._readout=None;state['failure']=None
 with pytest.raises(ValueError,match='Parent readout replaced'):model.review_response('label0 suitable words','ro')
 assert state['calls']==previous_calls and state['factory']==previous_factories


import threading

def test_default_other_thread_cannot_borrow_pending_batch(tmp_path,monkeypatch):
 model,parent,state,holders,originals,assets=make_model(tmp_path,monkeypatch,True)
 ready=threading.Event();release=threading.Event();returned=threading.Event();success=[];errors=[]
 def owner():
  try:
   with model.batch():
    assets['offset_theta'].write_bytes(b'corrupt while first thread owns pending batch')
    ready.set();assert release.wait(3)
  except BaseException as e:errors.append(('owner',e))
 def second():
  assert ready.wait(3)
  try:success.append(model.review_response('label0 suitable words','ro'))
  except BaseException as e:errors.append(('second',e))
  finally:returned.set()
 a=threading.Thread(target=owner);b=threading.Thread(target=second)
 a.start();assert ready.wait(3);b.start()
 try:returned.wait(.2)
 finally:release.set();a.join(3);b.join(3)
 assert not a.is_alive() and not b.is_alive()
 assert not success, 'Second default call returned before owning its full verification boundary'
 assert state['calls']==0
 assert any(who=='second' for who,e in errors)


def test_zero_public_path_refuses_already_bound_readout_replacement(tmp_path,monkeypatch):
 model,parent,state,holders,originals,_=make_model(tmp_path,monkeypatch,True)
 original=parent._readout;parent._readout=types.SimpleNamespace(adapters=original.adapters,logits=lambda features:[0.,0.,0.,9.,0.])
 with pytest.raises(ValueError,match='readout|Readout'):model.review_response('label0 suitable words','ro')
 assert state['calls']==0


def test_delegate_failure_restores_original72_and_readout(tmp_path,monkeypatch):
 model,parent,state,holders,originals,_=make_model(tmp_path,monkeypatch)
 original=parent._readout
 def fail(*args):raise RuntimeError('synthetic offset arithmetic failure')
 monkeypatch.setattr(rt,'adjusted_logits',fail)
 with pytest.raises(RuntimeError,match='offset arithmetic'):model.review_response_detailed('label0 suitable words','ro')
 assert state['calls']==1;restored(parent,original,holders,originals)


def test_float64_sub_float32_adjustment_changes_public_class(tmp_path,monkeypatch):
 import numpy as np
 save=np.save
 def save_tiny(file,array,*args,**kwargs):
  if Path(file).name=='theta.npy':
   array=array.copy();array[:]=0.;array[-4]=1e-12
  return save(file,array,*args,**kwargs)
 monkeypatch.setattr(np,'save',save_tiny)
 model,parent,state,holders,originals,_=make_model(tmp_path,monkeypatch)
 state['raw']=[1.,1.,-1.,-2.,-3.];readout=parent._readout
 d=model.review_response_detailed('label0 suitable words','ro')
 assert d.parent_logits[0]==d.parent_logits[1]==1.
 assert d.logits[1]>d.logits[0] and np.float32(d.logits[1])==np.float32(d.logits[0])
 assert d.response['label']=='boundary' and d.response['review_requested'] is True and state['calls']==1
 restored(parent,readout,holders,originals)


def test_adjusted_tie_uses_public_first_label_and_flags(tmp_path,monkeypatch):
 model,parent,state,holders,originals,_=make_model(tmp_path,monkeypatch)
 state['raw']=[0.,-3.,-1.,-2.,-3.];readout=parent._readout
 d=model.review_response_detailed('label0 suitable words','uk')
 assert d.logits[0]==d.logits[1]==0. and d.response['label']=='appropriate' and d.response['review_requested'] is False
 assert state['calls']==1;restored(parent,readout,holders,originals)
