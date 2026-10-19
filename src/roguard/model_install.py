"""Explicit, resumable installation of pinned optional model files."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shutil
import unicodedata
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request


def _https(url):
    if not isinstance(url,str):raise ValueError('Model URL must be HTTPS')
    value=urlsplit(url)
    if value.scheme!='https' or not value.hostname or value.username or value.password or value.fragment:
        raise ValueError('Model URL must be HTTPS without credentials or fragments')
    return url


class _Redirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        _https(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def _download(url):
    return build_opener(_Redirect()).open(Request(_https(url),headers={'User-Agent':'RoGuard model installer'}),timeout=60)


def _hash(value):
    if not isinstance(value,str) or re.fullmatch('[a-f0-9]{64}',value) is None:
        raise ValueError('Expected a SHA256 digest')
    return value


def _relative(value):
    if (not isinstance(value,str) or not value or unicodedata.normalize('NFC',value)!=value or '\\' in value or ':' in value
            or value.startswith('/') or any(p in ('','..','.') for p in value.split('/'))):
        raise ValueError('Model files must use normalized relative paths')
    return PurePosixPath(value)


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate manifest key')
        result[key]=value
    return result


def manifest(raw,expected_sha256):
    if (not isinstance(raw,bytes) or len(raw)>1_000_000
            or hashlib.sha256(raw).hexdigest()!=_hash(expected_sha256)):
        raise ValueError('Model manifest identity or size differs')
    value=json.loads(raw,object_pairs_hook=_unique)
    if (not isinstance(value,dict) or set(value)!={'format','id','bundle','files','terms'}
            or value['format']!='roguard-model-download-v1'
            or not isinstance(value['id'],str) or re.fullmatch('[a-z0-9][a-z0-9._-]{0,79}',value['id']) is None
            or not isinstance(value['files'],list) or not 1<=len(value['files'])<=256
            or not isinstance(value['terms'],list) or not value['terms']):
        raise ValueError('Unsupported model download manifest')
    for url in value['terms']:_https(url)
    files={};aliases=set();total=0
    for entry in value['files']:
        if not isinstance(entry,dict) or set(entry)!={'path','url','sha256','bytes'}:
            raise ValueError('Invalid model file declaration')
        _relative(entry['path']);_https(entry['url']);_hash(entry['sha256'])
        if type(entry['bytes']) is not int or not 1<=entry['bytes']<=10_000_000_000:
            raise ValueError('Invalid model file size')
        alias=entry['path'].casefold()
        if alias in aliases:raise ValueError('Duplicate or case-aliased model file path')
        aliases.add(alias)
        files[entry['path']]=entry;total+=entry['bytes']
    if total>20_000_000_000:raise ValueError('Model download exceeds size limit')
    # Reject file/directory collisions before any download or filesystem write.
    for name in files:
        if any(str(parent).casefold() in aliases for parent in PurePosixPath(name).parents):
            raise ValueError('Model file paths conflict')
    bundle=value['bundle']
    if (not isinstance(bundle,dict) or set(bundle)!={'path','sha256'}
            or bundle['path'] not in files or bundle['sha256']!=files[bundle['path']]['sha256']):
        raise ValueError('Bundle must identify a pinned downloaded file')
    return value


def _target(root,name):
    relative=_relative(name);path=root
    for part in relative.parts:
        path=path/part
        if path.is_symlink():raise ValueError('Model cache must not contain symlinks')
    if root.resolve() not in path.resolve().parents:raise ValueError('Model cache path escapes its directory')
    return path


def _matches(path,entry):
    if not path.is_file() or path.stat().st_size!=entry['bytes']:return False
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()==entry['sha256']


def install(raw,expected_sha256,cache,*,download=False,progress=None):
    spec=manifest(raw,expected_sha256);cache=Path(cache).expanduser().resolve()
    root=_target(cache,spec['id'])
    # Validate all paths first, including previously present symlinks.
    targets=[_target(root,e['path']) for e in spec['files']]
    verified={}
    for index,(entry,path) in enumerate(zip(spec['files'],targets),1):
        identity=(entry['sha256'],entry['bytes'])
        if _matches(path,entry):
            verified[identity]=path
            continue
        if not download:raise ValueError('Model files are missing or changed; run roguard-fetch fetch explicitly')
        path.parent.mkdir(parents=True,exist_ok=True)
        if _target(root,entry['path'])!=path:raise ValueError('Model cache path changed')
        if identity in verified:
            descriptor,temporary=tempfile.mkstemp(prefix=path.name+'.part-',dir=path.parent)
            os.close(descriptor)
            try:
                os.unlink(temporary)
                try:os.link(verified[identity],temporary)
                except OSError:shutil.copyfile(verified[identity],temporary)
                if not _matches(Path(temporary),entry):raise ValueError('Cached model content changed')
                _target(root,entry['path'])
                os.replace(temporary,path)
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
            verified[identity]=path
            continue
        if progress:progress(index,len(targets),entry['path'],entry['bytes'])
        descriptor,temporary=tempfile.mkstemp(prefix=path.name+'.part-',dir=path.parent)
        try:
            digest=hashlib.sha256();received=0
            with os.fdopen(descriptor,'wb') as output:
                with _download(entry['url']) as response:
                    for chunk in iter(lambda:response.read(1024*1024),b''):
                        received+=len(chunk)
                        if received>entry['bytes']:raise ValueError('Downloaded model exceeds declared size')
                        output.write(chunk);digest.update(chunk)
                if received!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:
                    raise ValueError('Downloaded model failed its checksum or size check')
                output.flush();os.fsync(output.fileno())
            if _target(root,entry['path'])!=path:raise ValueError('Model cache path changed')
            os.replace(temporary,path)
            verified[identity]=path
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    return _target(root,spec['bundle']['path']),spec['bundle']['sha256']


def profile(name):
    directory=Path(__file__).parent
    try:catalog=json.loads((directory/'model-profiles.json').read_text())
    except FileNotFoundError as error:raise ValueError('This build has no published model profile yet') from error
    if not isinstance(catalog,dict) or set(catalog)!={'profiles'} or not isinstance(catalog['profiles'],dict):
        raise ValueError('Invalid packaged model profile catalog')
    entry=catalog['profiles'].get(name)
    if (not isinstance(entry,dict) or set(entry)!={'manifest','sha256','runtime'}
            or entry['runtime'] not in ('portable','apple-silicon','apple-silicon-python312')):raise ValueError('Unknown model profile')
    if entry['runtime']=='apple-silicon-python312' and (platform.system()!='Darwin' or platform.machine()!='arm64' or sys.version_info<(3,12)):
        raise ValueError('This optional model profile requires native Apple Silicon Python 3.12 or later')
    if entry['runtime']=='apple-silicon' and (platform.system()!='Darwin' or platform.machine()!='arm64' or sys.version_info<(3,10)):
        raise ValueError('This optional model profile requires native Apple Silicon Python 3.10 or later')
    name=_relative(entry['manifest'])
    if len(name.parts)!=1:raise ValueError('Profile manifest must be a packaged file')
    raw=(directory/str(name)).read_bytes();manifest(raw,entry['sha256'])
    return raw,entry['sha256']


def cache_directory():return Path(os.environ.get('ROGUARD_MODEL_CACHE',str(Path.home()/'.cache/roguard/models')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('fetch','locate'))
    p.add_argument('--profile',default='v9');p.add_argument('--cache',type=Path,default=cache_directory())
    a=p.parse_args()
    try:
        raw,digest=profile(a.profile)
        if a.action=='fetch':
            print('Model terms: '+', '.join(manifest(raw,digest)['terms']),file=sys.stderr)
        progress=lambda i,n,path,size:print(f'[{i}/{n}] {path} ({size/1_000_000:.1f} MB)',file=sys.stderr)
        path,_=install(raw,digest,a.cache,download=a.action=='fetch',progress=progress)
        print(path)
    except (OSError,ValueError) as error:p.error(str(error))


if __name__=='__main__':main()
