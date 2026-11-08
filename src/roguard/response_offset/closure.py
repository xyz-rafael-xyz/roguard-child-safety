"""Portable file closure adapted from reviewed structured wrapperv5.
Only runtime assets belong here; actual research lineage is verified externally.
"""
from pathlib import Path
import hashlib,json,sys,platform,importlib,importlib.metadata,importlib.util
from roguard.assurance import digest
from roguard.gemma4_contract import FILES as BASE_FILES,VERSIONS as RUNTIME_VERSIONS

def runtime_asset_ids():
 return {'base:'+name for name in BASE_FILES}|{'runtime_source','closure_source','lexical_source','response59_spec','response59_training_manifest','response59_weights','offset_manifest','offset_theta','offset_lexicon','reference_spec','reference_weights'}

def valid_sha256(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def resolved_asset_path(value):
 # HF snapshot links resolve to immutable-content blobs; bind bytes, not host paths.
 path=Path(value).resolve(strict=True);require(path.is_file() and not path.is_symlink(),'Asset must resolve to a regular file');return path

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
  r=self._registration;require(r['format']=='roguard-response-offset-installed-closure-1' and r['qualified'] is False and r['scope'] in ('synthetic_fixture','installed_candidate'),'Invalid closure scope')
  require(isinstance(r['packages'],dict) and r['packages'] and isinstance(r['assets'],dict) and r['assets'],'Explicit packages and assets required')
  require(set(r['assets'])==set(r['required_asset_ids']),'Exact transitive asset inventory required')
  if r['scope']=='installed_candidate':
   require(r['registered'] is True and {'roguard','mlx_lm','mlx','numpy','transformers','tokenizers','huggingface_hub','safetensors'}<=set(r['packages']),'Registered installed runtime/dependency roots required')
   versions={p['distribution']:p['version'] for p in r['packages'].values()}
   require(all(versions.get(name)==version for name,version in RUNTIME_VERSIONS.items()),'Pinned runtime dependency versions required')
   expected=runtime_asset_ids()
   require(set(r['assets'])==expected and type(r['required_asset_ids']) is list and len(r['required_asset_ids'])==len(expected) and type(r['required_runtime_assets']) is list and len(r['required_runtime_assets'])==len(expected) and set(r['required_runtime_assets'])==expected and r['native_dependencies'],'Exact portable runtime assets and native closure required; research files belong to external lineage')
  self.verify();self.receipt=self._receipt();self.identity=digest(self.receipt)
 def _receipt(self):
  r=self._registration
  return dict(format='roguard-response-offset-closure-receipt-1',scope=r['scope'],research_provenance_verification='declared_digests_only_external_verification_required',packages={name:{'files':v['files'],'distribution':v.get('distribution'),'version':v.get('version')} for name,v in r['packages'].items()},assets={name:{k:v[k] for k in ('bytes','sha256')} for name,v in r['assets'].items()},asset_edges=r['asset_edges'],native_dependencies={name:{k:v[k] for k in ('bytes','sha256')} for name,v in r['native_dependencies'].items()},declared_supported_runtime_contract=r['declared_supported_runtime_contract'])
 def asset_path(self,name):return resolved_asset_path(self._registration['assets'][name]['path'])
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
    path=resolved_asset_path(b['path']) if group=='assets' else Path(b['path']);require(path.is_file() and not path.is_symlink() and canonical(file_record(path))==canonical({k:b[k] for k in ('bytes','sha256')}),'Asset bytes differ: '+name)
  for edge in r['asset_edges']:
   value=read(self.asset_path(edge['from']))
   for part in edge['keys']:value=value[part]
   require(type(value) is str and value==self.asset_sha(edge['to']),'Transitive manifest digest differs')
  if hasattr(self,'identity'):require(digest(self._receipt())==self.identity,'Closure object mutated')
  return True


