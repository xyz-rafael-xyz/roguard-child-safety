from test_experimental_structured import fixture,payload,w,np
import struct,tempfile,unittest
class FieldLogitTests(unittest.TestCase):
 def test_same_call_tuple_preserves_both_languages_routes_and_signed_zero(self):
  for lang in ('ro','uk'):
   with tempfile.TemporaryDirectory() as d:
    a,cc,c,_=fixture(d,active_language=lang);values=np.zeros(19,np.float32);values[0]=np.float32(-0.);values[18]=np.float32(-0.);cc[lang].z=tuple(map(float,values))
    for route in ('residual','original_v9'):
     cc[lang].route=route;before=len(cc[lang].calls);result=a.predict(payload(lang));self.assertEqual(len(cc[lang].calls),before+1);self.assertIs(result['field_logits'],cc[lang].z);self.assertEqual(type(result['field_logits']),tuple);self.assertEqual(struct.pack('<19f',*result['field_logits']),values.tobytes())
     with self.assertRaises(TypeError):result['field_logits'][0]=0.
 def test_native_unsupported_and_memory_overflow_have_no_logits(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d,routes={'ro':'original_v9_unsupported'});result=a.predict(payload());self.assertIsNone(result['field_logits']);self.assertIsNone(result['predicted'])
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d,mode='uniform');result=a.predict(payload(text='z'*1300));self.assertIsNone(result['field_logits']);self.assertEqual(cc['ro'].calls,[])
 def test_memory_success_keeps_final_logits_not_direct_unsupported(self):
  with tempfile.TemporaryDirectory() as d:
   a,cc,c,_=fixture(d);cc['ro'].router=lambda p:'original_v9_unsupported' if len(p['turns'][0]['text'])>1000 else 'residual';p=payload();p['turns'][0]['text']='O propoziție neutră. '*90;result=a.predict(p);self.assertEqual(len(cc['ro'].calls),2);self.assertIsNone(result['direct_unsupported_response']['predicted']);self.assertEqual(result['route'],'residual');self.assertIs(result['field_logits'],cc['ro'].z)
 def test_historical_six_head_argmax_is_not_coupled_world_map(self):
  for lang in ('ro','uk'):
   with tempfile.TemporaryDirectory() as d:
    a,cc,c,_=fixture(d,active_language=lang);cc[lang].z=tuple([0.]*19);result=a.predict(payload(lang));raw={};offset=0
    for field,classes in w.ONTOLOGY.items():
     logits=result['field_logits'][offset:offset+len(classes)];raw[field]=classes[max(range(len(classes)),key=lambda i:logits[i])];offset+=len(classes)
    world=w.WORLDS[max(range(432),key=lambda i:result['probabilities'][i])];self.assertNotEqual(raw,world);self.assertEqual(raw['assertion'],'AFFIRMED');self.assertEqual(world['assertion'],'RETRACTED');self.assertEqual(raw['relation'],'NEW');self.assertEqual(world['relation'],'RETRACTS')
if __name__=='__main__':unittest.main()
