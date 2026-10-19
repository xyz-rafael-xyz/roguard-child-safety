from io import BytesIO
import hashlib
import json
import pytest
from roguard import model_install as m


def specification(files=None):
    files=files or {'bundle.json':b'{"format":"test"}','model/weights.bin':b'fixed-weights'}
    value=dict(format='roguard-model-download-v1',id='test-v9',
        bundle=dict(path='bundle.json',sha256=hashlib.sha256(files['bundle.json']).hexdigest()),
        files=[dict(path=k,url='https://example.org/'+k,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in files.items()],
        terms=['https://example.org/terms'])
    return value,files


def encoded(spec):
    raw=json.dumps(spec).encode();return raw,hashlib.sha256(raw).hexdigest()


def test_install_verifies_and_resumes_without_network_for_complete_files(tmp_path,monkeypatch):
    spec,files=specification();calls=[]
    def download(url):
        name=url.removeprefix('https://example.org/');calls.append(name);return BytesIO(files[name])
    monkeypatch.setattr(m,'_download',download)
    path,digest=m.install(*encoded(spec),tmp_path,download=True)
    assert path.read_bytes()==files['bundle.json'] and digest==spec['bundle']['sha256']
    assert len(calls)==2
    m.install(*encoded(spec),tmp_path,download=True);m.install(*encoded(spec),tmp_path)
    assert len(calls)==2
    (path.parent/'model/weights.bin').write_bytes(b'bad')
    with pytest.raises(ValueError,match='missing or changed'):m.install(*encoded(spec),tmp_path)
    m.install(*encoded(spec),tmp_path,download=True)
    assert calls==['bundle.json','model/weights.bin','model/weights.bin']


@pytest.mark.parametrize('content',[b'wrong',b'fixed-weights-extra',b''])
def test_corrupt_or_interrupted_download_never_replaces_existing_file(tmp_path,monkeypatch,content):
    spec,files=specification();root=tmp_path/'test-v9';root.mkdir();(root/'bundle.json').write_bytes(files['bundle.json'])
    destination=root/'model/weights.bin';destination.parent.mkdir();destination.write_bytes(b'old')
    monkeypatch.setattr(m,'_download',lambda url:BytesIO(content))
    with pytest.raises(ValueError):m.install(*encoded(spec),tmp_path,download=True)
    assert destination.read_bytes()==b'old'
    assert not list(destination.parent.glob('*.part-*'))


@pytest.mark.parametrize('path',['../escape','/absolute','model/../escape','model\\escape','C:escape','./bundle','model//file'])
def test_unsafe_manifest_rejected_before_download_or_writes(tmp_path,monkeypatch,path):
    spec,_=specification();spec['files'][1]['path']=path
    monkeypatch.setattr(m,'_download',lambda url:pytest.fail('Downloaded invalid manifest'))
    with pytest.raises(ValueError):m.install(*encoded(spec),tmp_path,download=True)
    assert not list(tmp_path.iterdir())


def test_existing_symlink_cannot_redirect_model_write(tmp_path,monkeypatch):
    spec,_=specification();root=tmp_path/'test-v9';root.mkdir();outside=tmp_path/'outside';outside.mkdir()
    (root/'model').symlink_to(outside,target_is_directory=True)
    monkeypatch.setattr(m,'_download',lambda url:pytest.fail('Downloaded into symlink'))
    with pytest.raises(ValueError,match='symlinks'):m.install(*encoded(spec),tmp_path,download=True)
    assert not list(outside.iterdir())


@pytest.mark.parametrize('change',['duplicate','collision','size','url','bundle','hash'])
def test_malformed_manifest_cannot_start_download(change):
    spec,_=specification()
    if change=='duplicate':spec['files'].append(dict(spec['files'][0]))
    elif change=='collision':spec['files'][1]['path']='bundle.json/child'
    elif change=='size':spec['files'][1]['bytes']=True
    elif change=='url':spec['files'][1]['url']='http://example.org/file'
    elif change=='bundle':spec['bundle']['sha256']='f'*64
    else:spec['files'][1]['sha256']='bad'
    with pytest.raises(ValueError):m.manifest(*encoded(spec))


def test_manifest_checksum_is_required():
    spec,_=specification();raw,_=encoded(spec)
    with pytest.raises(ValueError):m.manifest(raw,'f'*64)


def test_identical_files_in_language_bundles_download_once(tmp_path,monkeypatch):
    spec,files=specification({'bundle.json':b'{}','ro/shared.bin':b'same','uk/shared.bin':b'same'})
    calls=[]
    def download(url):
        name=url.removeprefix('https://example.org/');calls.append(name);return BytesIO(files[name])
    monkeypatch.setattr(m,'_download',download)
    m.install(*encoded(spec),tmp_path,download=True)
    assert calls==['bundle.json','ro/shared.bin']
    assert (tmp_path/'test-v9/uk/shared.bin').read_bytes()==b'same'


def test_case_insensitive_filesystems_cannot_alias_manifest_paths():
    spec,_=specification();spec['files'][1]['path']='BUNDLE.JSON'
    with pytest.raises(ValueError):m.manifest(*encoded(spec))


def test_profile_reports_unsupported_runtime_before_reading_or_downloading_models(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'__file__',str(tmp_path/'model_install.py'))
    (tmp_path/'model-profiles.json').write_text(json.dumps(dict(profiles=dict(v9=dict(
        manifest='model-download-v9.json',sha256='a'*64,runtime='apple-silicon')))))
    monkeypatch.setattr(m.platform,'system',lambda:'Linux')
    with pytest.raises(ValueError,match='Apple Silicon'):m.profile('v9')


@pytest.mark.parametrize('system,machine,python', [('Linux','aarch64',(3,12)), ('Darwin','x86_64',(3,12)), ('Darwin','arm64',(3,11))])
def test_gemma4_profile_rejects_incompatible_platform_before_manifest_access(tmp_path,monkeypatch,system,machine,python):
    monkeypatch.setattr(m,'__file__',str(tmp_path/'model_install.py'))
    (tmp_path/'model-profiles.json').write_text(json.dumps(dict(profiles=dict(v9=dict(
        manifest='model-download-v9.json',sha256='a'*64,runtime='apple-silicon-python312')))))
    monkeypatch.setattr(m.platform,'system',lambda:system)
    monkeypatch.setattr(m.platform,'machine',lambda:machine)
    monkeypatch.setattr(m.sys,'version_info',python)
    with pytest.raises(ValueError,match='Apple Silicon Python 3.12'):m.profile('v9')


def test_gemma4_profile_accepts_pinned_manifest_on_supported_platform(tmp_path,monkeypatch):
    spec,_=specification();raw,identity=encoded(spec)
    monkeypatch.setattr(m,'__file__',str(tmp_path/'model_install.py'))
    (tmp_path/'model-download-test.json').write_bytes(raw)
    (tmp_path/'model-profiles.json').write_text(json.dumps(dict(profiles=dict(test=dict(
        manifest='model-download-test.json',sha256=identity,runtime='apple-silicon-python312')))))
    monkeypatch.setattr(m.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(m.platform,'machine',lambda:'arm64')
    monkeypatch.setattr(m.sys,'version_info',(3,12))
    assert m.profile('test')==(raw,identity)
