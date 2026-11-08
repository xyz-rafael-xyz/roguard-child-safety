from pathlib import Path
import importlib.util,json,copy,types,sys,tempfile,unittest,collections,math
import pytest
np=pytest.importorskip('numpy')
from roguard import experimental_structured as w
O=Path(__file__).parent
SPEC_FIXTURES=O/'fixtures/role-residual-specs'
Detail=collections.namedtuple('DetailedPrediction','response logits route spec_sha256')
class Selector:
 def __init__(self,identity):self.identity=identity;self.spec={'min_relevance':None}
 def relevance(self,text,language,role):return 1.
class Fake:
 def __init__(self,lang,closure,route='residual',router=None):self.lang=lang;self.c=closure;self.route=route;self.router=router;self.calls=[];self.failure=None;self.mutate=None;self.z=tuple(map(float,np.linspace(-1,1,19,dtype=np.float32)))
 def predict_detailed(self,payload,focus=None):
  self.calls.append(copy.deepcopy(payload))
  if self.failure:raise self.failure
  route=self.router(payload) if self.router else self.route;which='75' if route=='residual' else '9';spec=json.loads(self.c.asset_path(self.lang+'_spec'+which).read_text());z=None if route=='original_v9_unsupported' else self.z;score=None if z is None else float(w._score(z,spec));pred=None if score is None else int(score>=spec['threshold'])
  raw=dict(language=self.lang,focus=focus,input_digest=w.input_identity(payload),model_identity=self.c.asset_sha(self.lang+'_residual_manifest') if route=='residual' else w.SPEC9[self.lang],score=score,predicted=pred,decision='clarify' if pred is None else 'review' if pred else 'no_signal',review_requested=pred!=0)
  d=Detail(raw,z,route,self.c.asset_sha(self.lang+'_spec'+which));return self.mutate(d) if self.mutate else d

def fixture(root,mode='direct-first',routes=None,active_language='ro'):
 root=Path(root);package=root/'package';package.mkdir();(package/'__init__.py').write_text('# synthetic source\n');(package/'lazy.py').write_text('X=1\n');assets={}
 def add(name,data):
  p=root/(name+'.json');p.write_bytes(data);assets[name]=dict(path=str(p),**w.file_record(p))
 for lang in ('ro','uk'):
  for which,path in [('75',SPEC_FIXTURES/f'{lang}-75.json'),('9',SPEC_FIXTURES/f'{lang}-v9.json')]:add(lang+'_spec'+which,path.read_bytes())
  add(lang+'_residual_manifest',json.dumps({'synthetic':lang}).encode())
 add('selector',b'{"synthetic":"selector"}')
 add('leaf',b'{"synthetic":"leaf"}');add('parent',json.dumps({'child_sha':assets['leaf']['sha256']}).encode())
 registration=dict(format='roguard-structured-installed-closure-1',qualified=False,scope='synthetic_fixture',registered=False,packages={'synthetic_runtime':dict(root=str(package),files=w.python_inventory(package),distribution=None,version=None)},assets=assets,required_asset_ids=list(assets),asset_edges=[dict(**{'from':'parent'},keys=['child_sha'],to='leaf')],native_dependencies={},declared_supported_runtime_contract={'python_minimum':'3.12','platform':'AppleSilicon','native_backend':'MLX'})
 p=root/'closure.json';p.write_text(json.dumps(registration));c=w.Closure(p,w.sha(p));components={lang:Fake(lang,c,route=(routes or {}).get(lang,'residual')) for lang in ('ro','uk')};selector=Selector(c.asset_sha('selector'));wrapper=w.StructuredWrapper({active_language:components[active_language]},selector,c,active_language=active_language,mode=mode);return wrapper,components,c,registration

def payload(lang='ro',text='Acesta este mesajul meu.'):
 return {'language':lang,'turns':[{'role':'assistant','text':'Context neutru.'},{'role':'minor','text':text}]}
class Tests(unittest.TestCase):
 def test_coupled_bilingual_same_native_scores(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d)
   for lang in ('ro','uk'):
    a=w.StructuredWrapper({lang:cc[lang]},a.selector,c,active_language=lang)
    result=a.predict(payload(lang));self.assertEqual(result['route'],'residual');self.assertEqual(result['score'].hex(),result['native_response']['score'].hex());self.assertEqual(result['predicted'],result['native_response']['predicted']);self.assertAlmostEqual(sum(result['probabilities']),1,places=14);self.assertLessEqual(result['semantic_score_relation']['semantic_vs_binary_difference'],w.MASS_ATOL);self.assertNotIn('interpretation_set',result)
 def test_614_to1024_v9_fallback_never_selects_memory_and_uk_score_distinct(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d,routes={'ro':'original_v9','uk':'original_v9'})
   for lang in ('ro','uk'):
    a=w.StructuredWrapper({lang:cc[lang]},a.selector,c,active_language=lang)
    cc[lang].z=tuple(float(v) for v in np.zeros(19,np.float32));r=a.predict(payload(lang));self.assertFalse(r['used_memory']);self.assertEqual(len(cc[lang].calls),1);self.assertEqual(r['route'],'original_v9');self.assertEqual(r['semantic_score_relation']['score_mass_equivalence_required'],lang=='ro')
    if lang=='uk':self.assertGreater(r['semantic_score_relation']['semantic_vs_binary_difference'],.01)
 def test_unsupported_remains_fullontology_no_false_confidence(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d,routes={'ro':'original_v9_unsupported'});r=a.predict(payload());self.assertIsNone(r['score']);self.assertIsNone(r['predicted']);self.assertEqual(len(r['interpretation_set']['hypotheses']),432);self.assertEqual(r['probabilities_source'],'UNSUPPORTED_UNIFORM_PLACEHOLDER');self.assertEqual(len(cc['ro'].calls),2);self.assertTrue(r['used_memory'])
 def test_original_turn_overflow_selects_memory_before_native(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d);p=payload();p['turns']=[{'role':'assistant','text':'x'*180} for _ in range(13)]+[{'role':'minor','text':'Ultimul mesaj.'}];r=a.predict(p);self.assertTrue(r['used_memory']);self.assertEqual(len(cc['ro'].calls),1);self.assertLessEqual(len(cc['ro'].calls[0]['turns']),12);self.assertEqual(r['focus'],13);self.assertNotEqual(r['input_digest'],r['native_response']['input_digest'])
 def test_mandatory_overflow_never_drops_focal(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d,mode='uniform');r=a.predict(payload(text='a'*1300));self.assertEqual(r['route'],'memory_unsupported');self.assertEqual(r['evidence'],[]);self.assertEqual(len(r['interpretation_set']['hypotheses']),432);self.assertEqual(cc['ro'].calls,[])
 def test_unicode_original_offsets_and_focus_no_future(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d,mode='uniform');p=payload(text='  Eu va\u0306d 👩🏽‍💻.\nA doua propoziție!  ');p['turns'].append({'role':'minor','text':'Nu consulta viitorul.'});saved=copy.deepcopy(p);r=a.predict(p,focus=1);self.assertEqual(p,saved);self.assertEqual(r['focus'],1);self.assertTrue(all(s['turn']<=1 for s in r['evidence']));self.assertNotIn('Nu consulta',json.dumps(cc['ro'].calls));self.assertEqual(cc['ro'].calls[0]['turns'][-1]['text'],p['turns'][1]['text'])
   for seg in r['inference_to_original_segments']:
    source=p['turns'][seg['source_turn']]['text'][seg['source_start']:seg['source_end']];actual=cc['ro'].calls[0]['turns'][seg['inference_turn']]['text'][seg['inference_start']:seg['inference_end']];self.assertEqual(source,actual)
 def test_exceptions_and_invalid_detailed_records_propagate_without_fallback(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d);cc['ro'].failure=RuntimeError('synthetic native failure')
   with self.assertRaises(RuntimeError):a.predict(payload())
   self.assertEqual(len(cc['ro'].calls),1);cc['ro'].failure=None
   for mutate in (lambda d:d._replace(spec_sha256='0'*64),lambda d:d._replace(logits=list(d.logits)),lambda d:d._replace(response=dict(d.response,predicted=False)),lambda d:d._replace(response=dict(d.response,input_digest='0'*64))):
    cc['ro'].mutate=mutate
    with self.assertRaises(ValueError):a.predict(payload())
 def test_source_and_asset_corruptions_reject_before_any_calls(self):
  for kind in ('source','extra_source','asset','edge'):
   with tempfile.TemporaryDirectory() as d:
    a,cc,c,r=fixture(d)
    if kind=='source':(Path(d)/'package/lazy.py').write_text('X=2\n')
    elif kind=='extra_source':(Path(d)/'package/new_lazy.py').write_text('X=2\n')
    elif kind=='asset':c.asset_path('leaf').write_text('{}')
    else:
     c.asset_path('parent').write_text('{"child_sha":"bad"}');r['assets']['parent'].update(w.file_record(c.asset_path('parent')));p=Path(d)/'closure.json';p.write_text(json.dumps(r))
     with self.assertRaises(ValueError):w.Closure(p,w.sha(p))
     continue
    with self.assertRaises(ValueError):a.predict(payload())
    self.assertEqual(cc['ro'].calls,[])
 def test_source_import_shadow_and_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,r=fixture(d);foreign=types.ModuleType('synthetic_runtime.lazy');foreign.__file__=str(Path(d)/'leaf.json');sys.modules['synthetic_runtime.lazy']=foreign
   try:
    with self.assertRaises(ValueError):c.verify()
   finally:del sys.modules['synthetic_runtime.lazy']
   (Path(d)/'package/alias.py').symlink_to(Path(d)/'package/lazy.py')
   with self.assertRaises(ValueError):c.verify()
 def test_portable_identity_relocates_but_modes_differ_no_host_ids(self):
  with tempfile.TemporaryDirectory() as d1,tempfile.TemporaryDirectory() as d2:
   a,_,c,_=fixture(d1);b,_,_,_=fixture(d2);self.assertEqual(a.identity,b.identity);self.assertNotIn(d1,w.canonical(a.contract));self.assertNotIn(d2,w.canonical(b.contract));uniform=w.StructuredWrapper(a.components,a.selector,c,active_language=a.active_language,mode='uniform');self.assertNotEqual(a.identity,uniform.identity);self.assertNotIn('hostname',w.host_diagnostics())
 def test_calibration_mismatch_rejected_without_fitting(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d)
   with self.assertRaises((ValueError,KeyError)):a.predict(payload(),calibration={'pipeline_digest':'0'*64})
 def test_true_overflow_retries_only_selected_memory_and_retains_original(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d);cc['ro'].router=lambda p:'original_v9_unsupported' if len(p['turns'][0]['text'])>1000 else 'residual';p=payload();p['turns'][0]['text']='O propoziție neutră. '*90;r=a.predict(p);self.assertEqual(r['route'],'residual');self.assertEqual(len(cc['ro'].calls),2);self.assertEqual(cc['ro'].calls[0],p);self.assertEqual(r['input_digest'],w.input_identity(p));self.assertIsNone(r['direct_unsupported_response']['score']);self.assertTrue(r['used_memory'])
 def test_partial_unicode_segments_and_independent_mode_calibration(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d,mode='uniform');p=payload(text='Eu văd 👩🏽‍💻.');p['turns'][0]['text']='  '+('Text lung neutru '+('z'*125)+'.\n')*12;r=a.predict(p);self.assertTrue(any(s['source_start']>0 or s['source_end']<len(p['turns'][s['source_turn']]['text']) for s in r['inference_to_original_segments']))
   for seg in r['inference_to_original_segments']:
    self.assertEqual(p['turns'][seg['source_turn']]['text'][seg['source_start']:seg['source_end']],cc['ro'].calls[0]['turns'][seg['inference_turn']]['text'][seg['inference_start']:seg['inference_end']])
   # Construct an empty synthetic calibration-format artifact directly; no calibration fit.
   artifact=dict(format='roguard-joint-calibration-v10-research2',pipeline_digest=a.identity,policy_digest=w.policy_digest(),probability_floors={'ro':0.,'uk':0.});artifact['digest']=w.digest(artifact)
   self.assertEqual(len(a.predict(p,calibration=artifact)['interpretation_set']['hypotheses']),432)
   other=w.StructuredWrapper(a.components,a.selector,c,active_language=a.active_language,mode='direct-first')
   with self.assertRaises(ValueError):other.predict(payload(),calibration=artifact)
 def test_late_bound_asset_mutation_rejects_result(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d)
   def mutate(detail):c.asset_path('leaf').write_text('{}');return detail
   cc['ro'].mutate=mutate
   with self.assertRaises(ValueError):a.predict(payload())
 def test_owned_selector_and_result_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d);p=payload();r=a.predict(p);r['native_response']['score']=0.;p['turns'][-1]['text']='changed';self.assertNotEqual(a.predict(payload())['score'],0.);a.selector.spec['min_relevance']=.9
   with self.assertRaises(ValueError):a.predict(payload())
if __name__=='__main__':unittest.main()
