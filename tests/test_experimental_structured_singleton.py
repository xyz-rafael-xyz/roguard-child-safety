from test_experimental_structured import fixture,payload,w,Selector
from test_experimental_structured_installed import installed_fixture,runtime
from pathlib import Path
from unittest.mock import patch
import copy,importlib.util,json,tempfile,unittest
class SingletonTests(unittest.TestCase):
 def test_both_languages_share_complete_bilingual_identity(self):
  with tempfile.TemporaryDirectory() as d:
   ro,components,c,_=fixture(d);uk=w.StructuredWrapper({'uk':components['uk']},ro.selector,c,active_language='uk')
   self.assertEqual(ro.identity,uk.identity);self.assertEqual(ro.contract,uk.contract);self.assertEqual(set(ro.components),{'ro'});self.assertEqual(set(uk.components),{'uk'});self.assertNotIn('active_language',ro.contract)
   self.assertEqual(set(ro.specs),{'ro','uk'});self.assertEqual(set(uk.manifest_ids),{'ro','uk'})
   artifact=dict(format='roguard-joint-calibration-v10-research2',pipeline_digest=ro.identity,policy_digest=w.policy_digest(),probability_floors={'ro':0.,'uk':0.});artifact['digest']=w.digest(artifact)
   for a,lang in ((ro,'ro'),(uk,'uk')):
    result=a.predict(payload(lang),calibration=artifact);self.assertEqual(result['pipeline_digest'],ro.identity);self.assertEqual(result['language'],lang);self.assertEqual(len(result['interpretation_set']['hypotheses']),432)
 def test_singleton_outputs_preserve_v2_semantics(self):
  reference=json.loads((Path(__file__).parent/'fixtures/structured-v2-synthetic-reference.json').read_text())
  self.assertEqual(reference['source_sha256'],'c710dd29a9e524d75556b6a114a3348f0bb79c04612ecbd111e4d3cacd23cf8b')
  with tempfile.TemporaryDirectory() as d:
   a,components,c,_=fixture(d)
   for lang in ('ro','uk'):
    current=w.StructuredWrapper({lang:components[lang]},a.selector,c,active_language=lang)
    for route in ('residual','original_v9','original_v9_unsupported'):
     components[lang].route=route;actual=current.predict(payload(lang));expected=reference['records'][lang+':'+route]
     for key,expected_digest in expected.items():self.assertEqual(w.digest(actual[key]),expected_digest,(lang,route,key))
 def test_invalid_component_maps_reject_without_calls(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d)
   for components,active in (({},'ro'),(cc,'ro'),({'uk':cc['uk']},'ro'),({'ro':cc['ro']},'xx'),({'ro':cc['ro']},True)):
    with self.assertRaises(ValueError):w.StructuredWrapper(components,a.selector,c,active_language=active)
   self.assertEqual(cc['ro'].calls,[]);self.assertEqual(cc['uk'].calls,[])
 def test_inactive_language_rejects_before_selection_or_native(self):
  for mode in ('direct-first','uniform'):
   with tempfile.TemporaryDirectory() as d:
    a,cc,c,_=fixture(d,mode=mode)
    with patch.object(w,'prepare_memory',side_effect=AssertionError('memory must not run')) as memory:
     with self.assertRaises(w.InactiveLanguageError):a.predict(payload('uk'))
     with self.assertRaises(w.InactiveLanguageError):a.predict_many([{'payload':payload('ro')},{'payload':payload('uk')}])
     self.assertEqual(memory.call_count,0)
    self.assertEqual(cc['ro'].calls,[]);self.assertEqual(cc['uk'].calls,[])
 def test_streamed_inactive_language_marks_batch_failed(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d)
   with self.assertRaises(w.InactiveLanguageError):
    with a.batch() as batch:
     result=batch.predict(payload());batch.predict(payload('uk'))
   self.assertEqual(result['closure_verification']['state'],'FAILED_BATCH');self.assertEqual(cc['uk'].calls,[])
 def test_inactive_asset_remains_bound(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,r=fixture(d);asset=c.asset_path('uk_residual_manifest');asset.write_text('{"synthetic":"changed UK manifest"}')
   with self.assertRaises(ValueError):a.predict(payload())
   self.assertEqual(cc['ro'].calls,[])
   r['assets']['uk_residual_manifest'].update(w.file_record(asset));registration=Path(d)/'closure.json';registration.write_text(json.dumps(r));newc=w.Closure(registration,w.sha(registration));new=w.StructuredWrapper({'ro':cc['ro']},a.selector,newc,active_language='ro');self.assertNotEqual(new.identity,a.identity)
 def test_installed_singletons_use_same_identity_and_wrong_language_engine_rejects(self):
  with tempfile.TemporaryDirectory() as d,installed_fixture(d) as (cc,selector,c,r):
   with patch.object(runtime,'_from_local_files',side_effect=AssertionError('No native loader allowed')) as loader:
    ro=w.StructuredWrapper({'ro':cc['ro']},selector,c,active_language='ro');uk=w.StructuredWrapper({'uk':cc['uk']},selector,c,active_language='uk');self.assertEqual(ro.identity,uk.identity)
    with self.assertRaises(ValueError):w.StructuredWrapper({'uk':cc['ro']},selector,c,active_language='uk')
    self.assertEqual(loader.call_count,0)
 def test_active_language_or_component_mutation_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d)
   with self.assertRaises(AttributeError):a.active_language='uk'
   a._active_language='uk'
   with self.assertRaises(ValueError):a.predict(payload('uk'))
   self.assertEqual(cc['ro'].calls,[]);self.assertEqual(cc['uk'].calls,[])
if __name__=='__main__':unittest.main()
