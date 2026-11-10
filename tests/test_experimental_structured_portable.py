from test_experimental_structured import fixture,w
from test_experimental_structured_installed import installed_fixture
from pathlib import Path
import ast,copy,json,tempfile,unittest
class PortableTests(unittest.TestCase):
 def rewritten(self,d,r):
  p=Path(d)/'changed-closure.json';p.write_text(json.dumps(r));return w.Closure(p,w.sha(p))
 def test_installed_no_research_files_required_and_exact_runtime_inventory(self):
  with tempfile.TemporaryDirectory() as d,installed_fixture(d) as (components,selector,c,r):
   self.assertEqual(set(r['assets']),w.runtime_asset_ids())
   for lang in ('ro','uk'):
    for name in ('selected_checkpoint','fit_plan','feature_manifest','selection_proof'):self.assertFalse((Path(d)/(lang+'_'+name+'.json')).exists())
   a=w.StructuredWrapper({'ro':components['ro']},selector,c,active_language='ro');a._guard(full=True)
   self.assertEqual(c.receipt['research_provenance_verification'],'declared_digests_only_external_verification_required')
   self.assertNotIn(d,w.canonical(a.contract))
 def test_research_extra_and_missing_runtime_asset_rejected(self):
  for kind in ('research_extra','missing_runtime','missing_runtime_declaration','duplicate_asset_declaration','duplicate_runtime_declaration'):
   with tempfile.TemporaryDirectory() as d,installed_fixture(d) as (_,_,_,r):
    r=copy.deepcopy(r)
    if kind=='research_extra':
     p=Path(d)/'fit-plan.json';p.write_text('{}');r['assets']['ro_fit_plan']=dict(path=str(p),**w.file_record(p));r['required_asset_ids'].append('ro_fit_plan');r['required_runtime_assets'].append('ro_fit_plan')
    elif kind=='missing_runtime':
     del r['assets']['uk_residual_tensors'];r['required_asset_ids'].remove('uk_residual_tensors');r['required_runtime_assets'].remove('uk_residual_tensors')
    elif kind=='missing_runtime_declaration':r['required_runtime_assets'].remove('selector')
    elif kind=='duplicate_asset_declaration':r['required_asset_ids'].append('selector')
    else:r['required_runtime_assets'].append('selector')
    with self.assertRaises(ValueError):self.rewritten(d,r)
 def test_four_provenance_digests_required_typed_and_lower_hex(self):
  for key in w.PROVENANCE_KEYS:
   for invalid in (None,False,3,'A'*64,'z'*64,'a'*63):
    with self.subTest(key=key,invalid=invalid),tempfile.TemporaryDirectory() as d,installed_fixture(d) as (components,selector,c,r):
     p=c.asset_path('ro_residual_manifest');m=w.read(p)
     if invalid is None:del m['provenance'][key]
     else:m['provenance'][key]=invalid
     p.write_text(json.dumps(m));r['assets']['ro_residual_manifest'].update(w.file_record(p));new=self.rewritten(d,r)
     with self.assertRaisesRegex(ValueError,'Four declared research'):w.StructuredWrapper({'ro':components['ro']},selector,new,active_language='ro')
 def test_changed_declared_digest_changes_identity_without_research_file(self):
  with tempfile.TemporaryDirectory() as d,installed_fixture(d) as (components,selector,c,r):
   before=w.StructuredWrapper({'ro':components['ro']},selector,c,active_language='ro');p=c.asset_path('ro_residual_manifest');m=w.read(p);m['provenance']['selected_checkpoint_sha256']='a'*64;p.write_text(json.dumps(m));r['assets']['ro_residual_manifest'].update(w.file_record(p));new=self.rewritten(d,r)
   residual=components['ro']._parent.residual;residual._manifest=copy.deepcopy(m);residual.identity=new.asset_sha('ro_residual_manifest')
   after=w.StructuredWrapper({'ro':components['ro']},selector,new,active_language='ro');self.assertNotEqual(before.identity,after.identity)
 def test_hf_snapshot_symlink_resolution_same_bytes_and_identity(self):
  with tempfile.TemporaryDirectory() as d:
   _,_,c,r=fixture(d);record=r['assets']['leaf'];blob=Path(record['path']);link=Path(d)/'snapshot-link.json';link.symlink_to(blob);r['assets']['leaf']['path']=str(link);new=self.rewritten(d,r)
   self.assertEqual(new.asset_path('leaf'),blob.resolve());self.assertEqual(new.identity,c.identity);new.verify()
   blob.write_text('{"changed":true}')
   with self.assertRaises(ValueError):new.verify()
 def test_symlink_dangling_loop_directory_or_wrong_bytes_rejected(self):
  for kind in ('dangling','loop','directory','wrong_bytes'):
   with tempfile.TemporaryDirectory() as d:
    _,_,_,r=fixture(d);link=Path(d)/'snapshot-link.json'
    if kind=='dangling':link.symlink_to(Path(d)/'missing')
    elif kind=='loop':link.symlink_to(link)
    elif kind=='directory':link.symlink_to(Path(d)/'package')
    else:
     wrong=Path(d)/'wrong';wrong.write_text('wrong');link.symlink_to(wrong)
    r['assets']['leaf']['path']=str(link)
    with self.assertRaises((ValueError,FileNotFoundError,RuntimeError,OSError)):self.rewritten(d,r)
 def test_numerical_routing_batch_methods_ast_unchanged(self):
  here=Path(__file__).parent;reference=json.loads((here/'fixtures/structured-v4-ast-reference.json').read_text());new=ast.parse(Path(w.__file__).read_text())
  def selected(tree):
   result={}
   for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ('coupled_worlds','check_evidence','rendering_segments'):result[node.name]=ast.dump(node,include_attributes=False)
    elif isinstance(node,ast.ClassDef) and node.name=='StructuredWrapper':
     for method in node.body:
      if isinstance(method,ast.FunctionDef) and method.name!='__init__':result[node.name+'.'+method.name]=ast.dump(method,include_attributes=False)
    elif isinstance(node,ast.ClassDef) and node.name=='VerifiedBatch':result[node.name]=ast.dump(node,include_attributes=False)
   return result
  actual={key:w.hashlib.sha256(value.encode()).hexdigest() for key,value in selected(new).items()};self.assertEqual(actual,reference['ast_blocks']);self.assertEqual(len(actual),14)
if __name__=='__main__':unittest.main()
