"""Installed-like closure branches, tiny test-only assets, no model construction.
Installed package discovery and immutable large asset pins are explicitly patched.
The reviewed real public Python classes are used with object.__new__ fixtures;
no real tensors, encoder or public loader is invoked.
"""
from test_experimental_structured import fixture,w,Selector,Fake
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager,ExitStack
import gzip,importlib.util,json,types,sys,tempfile,unittest,copy
from roguard import experimental_role_residual as runtime
W=Path(runtime.__file__).parent
@contextmanager
def installed_fixture(directory):
 root=Path(directory);_,_,_,r=fixture(root);r['scope']='installed_candidate';r['registered']=True
 assets=r['assets']
 for name in ('leaf','parent'):del assets[name]
 r['asset_edges']=[]
 def add(name,content):
  p=root/(name.replace(':','_')+'.json');p.write_bytes(content);assets[name]=dict(path=str(p),**w.file_record(p))
 add('selector',gzip.compress(json.dumps({'min_relevance':None}).encode(),mtime=0))
 selector=object.__new__(w.TrajectoryModel);selector.identity=assets['selector']['sha256'];selector.spec={'min_relevance':None}
 packages={};mapping={'roguard':'roguard-child-safety','mlx_lm':'mlx-lm','mlx':'mlx','numpy':'numpy','transformers':'transformers','tokenizers':'tokenizers','huggingface_hub':'huggingface-hub','safetensors':'safetensors'}
 for name,distribution in mapping.items():
  p=root/name;p.mkdir();(p/'__init__.py').write_text('# installed-like fixture\n')
  if name=='roguard':
   for filename in ('experimental_role_residual.py','role_alignment.py'):(p/filename).write_bytes((W/filename).read_bytes())
  packages[name]=dict(root=str(p),distribution=distribution,version=w.RUNTIME_VERSIONS.get(distribution,'0.fixture'),files=w.python_inventory(p))
 r['packages']=packages
 binary=root/'native_fixture.so';binary.write_bytes(b'synthetic binary, never imported');r['native_dependencies']={'synthetic_native':dict(path=str(binary),**w.file_record(binary))}
 pins75={};pins9={};components={}
 for lang in ('ro','uk'):
  for name in ('weights75','adapter9','residual_tensors'):add(lang+'_'+name,('synthetic '+lang+name).encode())
  pins75[lang]=assets[lang+'_weights75']['sha256'];pins9[lang]=assets[lang+'_adapter9']['sha256']
  manifest=dict(language=lang,tensor_sha256=assets[lang+'_residual_tensors']['sha256'],runtime_source_sha256=packages['roguard']['files']['experimental_role_residual.py']['sha256'],alignment_source_sha256=packages['roguard']['files']['role_alignment.py']['sha256'],parent={'original_v9_adapter_sha256':pins9[lang],'original_v9_spec_sha256':w.SPEC9[lang],'head75_source_sha256':pins75[lang],'spec_sha256':w.SPEC75[lang]},provenance={key:w.hashlib.sha256(('synthetic uninstalled research '+lang+key).encode()).hexdigest() for key in w.PROVENANCE_KEYS})
  add(lang+'_residual_manifest',json.dumps(manifest).encode())
  residual=object.__new__(runtime.ResidualArrays);residual._manifest=copy.deepcopy(manifest);residual.identity=assets[lang+'_residual_manifest']['sha256']
  engine=object.__new__(runtime.OptionalResidualEngine);engine.enabled=True;engine.residual=residual;engine.spec=w.read(assets[lang+'_spec75']['path']);engine.v9spec=w.read(assets[lang+'_spec9']['path']);engine.spec_identity=w.digest(engine.spec);engine.v9_spec_identity=w.digest(engine.v9spec);engine.alignment=types.SimpleNamespace(__file__=str(root/'roguard/role_alignment.py'))
  component=object.__new__(runtime.ExperimentalRoleResidualModel);component.enabled=True;component._parent=engine;components[lang]=component
 add('base:tiny.bin',b'tiny frozen synthetic backbone placeholder');base={'tiny.bin':{k:assets['base:tiny.bin'][k] for k in ('bytes','sha256')}};r['required_asset_ids']=list(assets);r['required_runtime_assets']=list(assets)
 registration=root/'closure.json';registration.write_text(json.dumps(r))
 removed={name:module for name,module in sys.modules.items() if any(name==p or name.startswith(p+'.') for p in packages)}
 for name in removed:del sys.modules[name]
 original_import=w.importlib.import_module
 try:
  with ExitStack() as stack:
   stack.enter_context(patch.object(w,'PIN75W',pins75));stack.enter_context(patch.object(w,'PIN9W',pins9));stack.enter_context(patch.object(w,'BASE_FILES',base));stack.enter_context(patch.object(w,'SELECTOR_SHA',assets['selector']['sha256']))
   def dist(name):
    p=next(p for p in packages.values() if p['distribution']==name);return types.SimpleNamespace(version=p['version'],locate_file=lambda _:root)
   stack.enter_context(patch.object(w.importlib.metadata,'distribution',side_effect=dist));stack.enter_context(patch.object(w.importlib.util,'find_spec',side_effect=lambda name:types.SimpleNamespace(submodule_search_locations=[packages[name]['root']])));stack.enter_context(patch.object(w.importlib,'import_module',side_effect=lambda name:runtime if name=='roguard.experimental_role_residual' else original_import(name)))
   closure=w.Closure(registration,w.sha(registration));yield components,selector,closure,r
 finally:
  for name in list(sys.modules):
   if any(name==p or name.startswith(p+'.') for p in packages):del sys.modules[name]
  sys.modules.update(removed)
class InstalledTests(unittest.TestCase):
 def test_installed_like_complete_closure_and_real_reviewed_types(self):
  with tempfile.TemporaryDirectory() as d,installed_fixture(d) as (components,selector,c,r):
   a=w.StructuredWrapper({'ro':components['ro']},selector,c,active_language='ro');a._guard(full=True);self.assertEqual(a.closure.receipt['scope'],'installed_candidate');self.assertNotIn(d,w.canonical(a.contract))
 def test_wrong_class_enable_identity_and_language_rejected(self):
  for kind in ('class','disabled','engine','identity','language','alignment','selector_class','selector_spec','selector_override'):
   with tempfile.TemporaryDirectory() as d,installed_fixture(d) as (components,selector,c,r):
    if kind=='class':components['ro']=Fake('ro',c)
    elif kind=='disabled':components['ro'].enabled=False
    elif kind=='engine':components['ro']._parent=types.SimpleNamespace(enabled=True)
    elif kind=='identity':components['ro']._parent.residual.identity='0'*64
    elif kind=='language':components['ro']._parent.spec['language']='uk'
    elif kind=='alignment':components['ro']._parent.alignment.__file__=str(Path(d)/'different.py')
    elif kind=='selector_class':selector=Selector(c.asset_sha('selector'))
    elif kind=='selector_spec':selector.spec['min_relevance']=.8
    else:selector.relevance=lambda *args:1.
    with self.assertRaises(ValueError):w.StructuredWrapper({'ro':components['ro']},selector,c,active_language='ro')
 def test_real_closure_constructor_rejects_provenance_version_and_native_bytes(self):
  for kind in ('manifest','version','native','missing_package'):
   with tempfile.TemporaryDirectory() as d,installed_fixture(d) as (components,selector,c,r):
    if kind=='manifest':
     p=c.asset_path('ro_residual_manifest');manifest=w.read(p);manifest['provenance']['selection_proof_sha256']='0'*64;p.write_text(json.dumps(manifest));r['assets']['ro_residual_manifest'].update(w.file_record(p))
    elif kind=='version':r['packages']['numpy']['version']='wrong'
    elif kind=='native':Path(r['native_dependencies']['synthetic_native']['path']).write_bytes(b'corrupt')
    else:del r['packages']['tokenizers']
    p=Path(d)/'closure.json';p.write_text(json.dumps(r))
    with self.assertRaises(ValueError):
     other=w.Closure(p,w.sha(p));w.StructuredWrapper({'ro':components['ro']},selector,other,active_language='ro')
if __name__=='__main__':unittest.main()
