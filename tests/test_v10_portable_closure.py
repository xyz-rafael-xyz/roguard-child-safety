import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import roguard.v10_closure as m


def record(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    roots = {}
    packages = {}
    for name in sorted(m.PACKAGES):
        root = tmp_path/'site'/name; root.mkdir(parents=True)
        raw = ('# synthetic '+name+'\n').encode(); (root/'__init__.py').write_bytes(raw)
        roots[name] = root
        packages[name] = dict(distribution=name, version='1.0', files={'__init__.py': record(raw)})
    for relative in m.HELPERS.values():
        path = roots['roguard']/relative; path.parent.mkdir(exist_ok=True)
        raw = ('# synthetic '+relative+'\n').encode(); path.write_bytes(raw)
        packages['roguard']['files'][relative] = record(raw)
    native = b'synthetic-native'; (roots['numpy']/'core.so').write_bytes(native)
    bundle = tmp_path/'bundle'; bundle.mkdir()
    original_find = m.importlib.util.find_spec
    original_dist = m.importlib.metadata.distribution
    monkeypatch.setattr(m.importlib.util, 'find_spec', lambda name: SimpleNamespace(submodule_search_locations=[str(roots[name])]) if name in roots else original_find(name))
    monkeypatch.setattr(m.importlib.metadata, 'distribution', lambda name: SimpleNamespace(version='1.0', locate_file=lambda p: tmp_path/'site'/p) if name in roots else original_dist(name))
    monkeypatch.setattr(m, 'sys', SimpleNamespace(modules={}))

    def make(kind='structured'):
        assets = {}
        for i, name in enumerate(sorted(m._asset_ids(kind))):
            if kind == 'response' and name in m.HELPERS:
                relative = m.HELPERS[name]; rec = packages['roguard']['files'][relative]
                assets[name] = dict(location=dict(root='package:roguard', relative_path=relative), **rec)
            else:
                relative = 'asset-'+str(i)+'.json'; raw = b'{}\n'; (bundle/relative).write_bytes(raw)
                assets[name] = dict(location=dict(root='bundle', relative_path=relative), **record(raw))
        value = dict(format=m.FORMAT, kind=kind, packages=copy.deepcopy(packages), assets=assets,
            native_dependencies={'numpy_core': dict(package='numpy', relative_path='core.so', **record(native))},
            asset_edges=[], declared_supported_runtime_contract={'synthetic': True}, expected_receipt_sha256='0'*64)
        return value

    def freeze(value):
        # Receipt ignores location fields, exactly as the reviewed classes do.
        value['expected_receipt_sha256'] = m.digest(m.receipt(value, value['kind']))
        path = tmp_path/'approved.json'; raw=(m.canonical(value)+'\n').encode(); path.write_bytes(raw)
        return path, record(raw)['sha256']
    return roots, bundle, make, freeze


@pytest.mark.parametrize('kind', ['structured', 'response'])
def test_success_exact_existing_receipt_and_relocation(fixture, tmp_path, kind):
    roots,bundle,make,freeze=fixture; value=make(kind); source,sha=freeze(value)
    binding=m.rebase_closure(source,expected_sha256=sha,bundle_root=bundle,output_path=tmp_path/'local.json')
    registered=json.loads(Path(binding.path).read_bytes())
    assert binding.receipt_sha256 == value['expected_receipt_sha256']
    # Evaluate the actual existing receipt method's AST without importing or
    # constructing models/dependencies or changing its implementation.
    w=Path(m.__file__).resolve().parent
    path=w/('experimental_structured.py' if kind=='structured' else 'response_offset/closure.py')
    tree=ast.parse(path.read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Closure')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_receipt')
    space={}; exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),space)
    existing=space['_receipt'](SimpleNamespace(_registration=registered))
    assert m.digest(existing)==binding.receipt_sha256
    moved=copy.deepcopy(registered)
    for entry in moved['packages'].values(): entry['root']='/different/installed/root'
    for group in ('assets','native_dependencies'):
        for entry in moved[group].values(): entry['path']='/different/asset'
    assert m.digest(m.receipt(moved,kind))==binding.receipt_sha256
    assert registered['qualified'] is False and registered['scope']=='installed_candidate'


@pytest.mark.parametrize('corruption', ['extra_asset','escape','absolute','helper_bundle','python_native','bad_receipt'])
def test_bad_declarations_reject_before_discovery(fixture,tmp_path,monkeypatch,corruption):
    _,bundle,make,freeze=fixture; value=make('response')
    if corruption=='extra_asset':value['assets']['TEST']=copy.deepcopy(value['assets']['offset_theta'])
    elif corruption=='escape':value['assets']['offset_theta']['location']['relative_path']='../TEST.json'
    elif corruption=='absolute':value['packages']['roguard']['files']['/TEST.py']=record(b'')
    elif corruption=='helper_bundle':value['assets']['runtime_source']['location']['root']='bundle'
    elif corruption=='python_native':value['native_dependencies']['numpy_core']['relative_path']='hidden.py'
    source,sha=freeze(value)
    if corruption=='bad_receipt':
        value['expected_receipt_sha256']='0'*64;source.write_text(m.canonical(value));sha=record(source.read_bytes())['sha256']
    if corruption!='bad_receipt':monkeypatch.setattr(m,'_roots',lambda _:pytest.fail('Discovery before declaration rejection'))
    with pytest.raises(ValueError):m.rebase_closure(source,expected_sha256=sha,bundle_root=bundle,output_path=tmp_path/'local.json')
    assert not (tmp_path/'local.json').exists()


@pytest.mark.parametrize('corruption',['package_bytes','extra_python','native_bytes','asset_bytes','version','loaded_origin'])
def test_actual_inventory_mismatch(fixture,tmp_path,monkeypatch,corruption):
    roots,bundle,make,freeze=fixture;value=make();source,sha=freeze(value)
    if corruption=='package_bytes':(roots['roguard']/'__init__.py').write_bytes(b'changed')
    elif corruption=='extra_python':(roots['roguard']/'injected.py').write_text('pass')
    elif corruption=='native_bytes':(roots['numpy']/'core.so').write_bytes(b'changed')
    elif corruption=='asset_bytes':(bundle/value['assets']['selector']['location']['relative_path']).write_bytes(b'changed')
    elif corruption=='version':monkeypatch.setattr(m.importlib.metadata,'distribution',lambda _:SimpleNamespace(version='2.0',locate_file=lambda p:tmp_path/'site'/p))
    else:monkeypatch.setattr(m,'sys',SimpleNamespace(modules={'roguard':SimpleNamespace(__file__=str(tmp_path/'foreign.py'))}))
    with pytest.raises(ValueError):m.rebase_closure(source,expected_sha256=sha,bundle_root=bundle,output_path=tmp_path/'local.json')
    assert not (tmp_path/'local.json').exists()


def test_outside_symlink_never_reads_target(fixture,tmp_path,monkeypatch):
    _,bundle,make,freeze=fixture;value=make();name=value['assets']['selector']['location']['relative_path']
    sentinel=tmp_path/'unopened-TEST';sentinel.write_bytes(b'{}\n');(bundle/name).unlink();(bundle/name).symlink_to(sentinel)
    source,sha=freeze(value);old=m._check_file
    def check(path,expected):
        assert path!=sentinel,'Outside target bytes accessed'
        return old(path,expected)
    monkeypatch.setattr(m,'_check_file',check)
    with pytest.raises(ValueError,match='escapes'):m.rebase_closure(source,expected_sha256=sha,bundle_root=bundle,output_path=tmp_path/'local.json')


def test_internal_bundle_symlink_allowed(fixture,tmp_path):
    _,bundle,make,freeze=fixture;value=make();name=value['assets']['selector']['location']['relative_path']
    target=bundle/'blob';target.write_bytes((bundle/name).read_bytes());(bundle/name).unlink();(bundle/name).symlink_to(target)
    source,sha=freeze(value)
    binding=m.rebase_closure(source,expected_sha256=sha,bundle_root=bundle,output_path=tmp_path/'local.json')
    assert json.loads(Path(binding.path).read_bytes())['assets']['selector']['path']==str(target)


def test_digest_edge_is_checked(fixture,tmp_path):
    _,bundle,make,freeze=fixture;value=make();entry=value['assets']['selector'];path=bundle/entry['location']['relative_path']
    raw=b'{"target":"wrong"}';path.write_bytes(raw);entry.update(record(raw));value['asset_edges']=[{'from':'selector','keys':['target'],'to':'ro_spec75'}]
    source,sha=freeze(value)
    with pytest.raises(ValueError,match='Transitive'):m.rebase_closure(source,expected_sha256=sha,bundle_root=bundle,output_path=tmp_path/'local.json')


def test_late_input_mutation_rejects_return(fixture,tmp_path,monkeypatch):
    roots,bundle,make,freeze=fixture;source,sha=freeze(make());old=m._verify;calls=[]
    def verify(*args):
        calls.append(1)
        if len(calls)==2:(roots['numpy']/'core.so').write_bytes(b'bad')
        return old(*args)
    monkeypatch.setattr(m,'_verify',verify)
    with pytest.raises(ValueError,match='size'):m.rebase_closure(source,expected_sha256=sha,bundle_root=bundle,output_path=tmp_path/'local.json')
    assert len(calls)==2


def test_late_resolved_asset_symlink_rejects_before_read(fixture,tmp_path,monkeypatch):
    _,bundle,make,freeze=fixture;value=make();source,sha=freeze(value)
    target=bundle/value['assets']['selector']['location']['relative_path']
    sentinel=tmp_path/'unopened-late-TEST';sentinel.write_bytes(b'{}\n')
    old=m._verify;calls=[]
    def verify(*args):
        calls.append(1)
        if len(calls)==2:target.unlink();target.symlink_to(sentinel)
        return old(*args)
    check=m._check_file
    def guarded(path,expected):
        assert path.resolve()!=sentinel,'Outside sentinel was opened'
        return check(path,expected)
    monkeypatch.setattr(m,'_verify',verify);monkeypatch.setattr(m,'_check_file',guarded)
    with pytest.raises(ValueError,match='symlink'):m.rebase_closure(source,expected_sha256=sha,bundle_root=bundle,output_path=tmp_path/'local.json')


def test_output_symlink_parent_cannot_modify_package(fixture,tmp_path):
    roots,bundle,make,freeze=fixture;source,sha=freeze(make());link=tmp_path/'linked';link.symlink_to(roots['roguard'],target_is_directory=True)
    with pytest.raises(ValueError,match='modify installed'):m.rebase_closure(source,expected_sha256=sha,bundle_root=bundle,output_path=link/'local.json')
    assert not (roots['roguard']/'local.json').exists()


def test_no_overwrite_and_approved_hash(fixture,tmp_path):
    _,bundle,make,freeze=fixture;source,sha=freeze(make());out=tmp_path/'local.json';out.write_text('retain')
    with pytest.raises(ValueError,match='digest'):m.rebase_closure(source,expected_sha256='0'*64,bundle_root=bundle,output_path=out)
    with pytest.raises(ValueError,match='overwrite'):m.rebase_closure(source,expected_sha256=sha,bundle_root=bundle,output_path=out)
    assert out.read_text()=='retain'
