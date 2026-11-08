from test_experimental_structured import fixture,payload,w,Fake
from pathlib import Path
import tempfile,unittest,types,sys
from unittest.mock import patch
class BatchTests(unittest.TestCase):
 def test_full_verification_only_two_boundaries(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d)
   with patch.object(c,'verify',wraps=c.verify) as verify:
    results=a.predict_many([dict(payload=payload()) for _ in range(5)])
    self.assertEqual(verify.call_count,2);self.assertEqual(len(cc['ro'].calls),5);self.assertTrue(all(r['closure_verification']['state']=='PASS' for r in results))
 def test_pending_results_close_and_nested_refusal(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d)
   with a.batch() as batch:
    result=batch.predict(payload());self.assertEqual(result['closure_verification']['state'],'PENDING_BATCH_CLOSE')
    with self.assertRaises(ValueError):a.predict(payload())
   self.assertEqual(result['closure_verification']['state'],'PASS')
   with self.assertRaises(ValueError):batch.predict(payload())
   with self.assertRaises(ValueError):batch.__enter__()
 def test_late_corruption_marks_streamed_results_failed(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d);result=None
   with self.assertRaises(ValueError):
    with a.batch() as batch:
     result=batch.predict(payload());c.asset_path('leaf').write_text('{}')
   self.assertEqual(result['closure_verification']['state'],'FAILED_BATCH');self.assertIsNone(a._active_batch)
 def test_perrow_configuration_and_module_origin_guards(self):
  for kind in ('configuration','origin','component'):
   with tempfile.TemporaryDirectory() as d:
    a,cc,c,_=fixture(d)
    try:
     with self.assertRaises((ValueError,ExceptionGroup)):
      with a.batch() as batch:
       batch.predict(payload())
       if kind=='configuration':a.selector.spec['min_relevance']=.1
       elif kind=='component':a.components['ro']=Fake('ro',c)
       else:
        module=types.ModuleType('synthetic_runtime.shadow');module.__file__=str(Path(d)/'leaf.json');sys.modules[module.__name__]=module
       batch.predict(payload())
     self.assertEqual(len(cc['ro'].calls),1)
    finally:sys.modules.pop('synthetic_runtime.shadow',None)
 def test_caught_row_failure_cannot_create_successful_batch(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d)
   with self.assertRaises(ValueError):
    with a.batch() as batch:
     result=batch.predict(payload());cc['ro'].failure=RuntimeError('caught row failure')
     with self.assertRaises(RuntimeError):batch.predict(payload())
   self.assertEqual(result['closure_verification']['state'],'FAILED_BATCH')
 def test_original_and_closing_failures_are_both_retained(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d)
   with self.assertRaises(ExceptionGroup) as caught:
    with a.batch() as batch:
     c.asset_path('leaf').write_text('{}');cc['ro'].failure=RuntimeError('original inference exception');batch.predict(payload())
   self.assertEqual(len(caught.exception.exceptions),2);self.assertIsInstance(caught.exception.exceptions[0],RuntimeError);self.assertIsInstance(caught.exception.exceptions[1],ValueError);self.assertIsNone(a._active_batch)
if __name__=='__main__':unittest.main()
