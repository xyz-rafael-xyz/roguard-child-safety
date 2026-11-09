"""Exercise actual roguard CLI modules; synthetic API doubles, no model assets."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from unittest import mock
import pytest

def package_source(name):
    import roguard
    root = Path(roguard.__file__).resolve().parent
    expected = os.environ.get('ROGUARD_TEST_EXPECTED_PACKAGE_ROOT')
    if expected:
        assert root == Path(expected).resolve(), 'Tests imported the wrong package installation'
    return root/name


@pytest.fixture
def cli(tmp_path, monkeypatch):
    import roguard.trajectory_cli as module
    assert Path(module.__file__).resolve() == package_source('trajectory_cli.py')
    calls=[]
    class Model:
        def predict(self, payload, **kwargs):
            calls.append(('predict', payload, kwargs))
            return {'qualified':False,'experimental':True,'probabilities':[0.25,0.75]}
        def review_response(self, draft, language, *, context):
            calls.append(('response',draft,language,context)); return {'qualified':False,'label':'appropriate'}
        def close(self): calls.append(('close',))
    model=Model()
    def load(path, *, expected_sha256): calls.append(('load',str(path),expected_sha256)); return model
    bundle=types.ModuleType('roguard.v10_bundle');bundle.load_v10=load
    monkeypatch.setitem(sys.modules,bundle.__name__,bundle)
    payload={'language':'ro','turns':[{'role':'minor','text':'Do not infer this as response context.'}]}
    path=tmp_path/'input.json';path.write_text(json.dumps(payload))
    return module, model, calls, path, payload

def args(path): return [str(path),'--bundle','manifest.json','--bundle-sha256','a'*64]

def test_bundle_explicit_context_and_flags(cli,capsys):
    module,model,calls,path,payload=cli
    module.main(args(path)+['--draft','A reply','--context','Caller declaration'])
    result=json.loads(capsys.readouterr().out)
    assert result=={'qualified':False,'experimental':True,'probabilities':[0.25,0.75],'response_review':{'qualified':False,'label':'appropriate'}}
    assert calls[1]==('predict',payload,{'calibration':None})
    assert calls[2]==('response','A reply','ro','Caller declaration')
    assert calls[-1]==('close',)

def test_explicit_empty_context(cli):
    module,model,calls,path,payload=cli
    module.main(args(path)+['--draft','Reply','--context',''])
    assert ('response','Reply','ro','') in calls

@pytest.mark.parametrize('extra',[['--neural'],['--plan'],['--model','other'],['--model-sha256','b'*64],['--draft','Reply'],['--context','alone']])
def test_conflicts_fail_before_loading(cli,extra):
    module,model,calls,path,payload=cli
    with pytest.raises(SystemExit) as e: module.main(args(path)+extra)
    assert e.value.code==2 and not calls

@pytest.mark.parametrize('flags',[['--bundle','x'],['--model','x'],['--model','x','--model-sha256','a'*64,'--bundle-sha256','b'*64]])
def test_missing_or_crossed_checksum(cli,flags):
    module,model,calls,path,payload=cli
    with pytest.raises(SystemExit) as e: module.main([str(path)]+flags)
    assert e.value.code==2 and not calls

@pytest.mark.parametrize('raw',['{','[]','x'*4_000_001],ids=['malformed','array','oversize'])
def test_bad_input_before_loader(cli,raw):
    module,model,calls,path,payload=cli;path.write_text(raw)
    with pytest.raises(SystemExit) as e: module.main(args(path))
    assert e.value.code==2 and not calls

@pytest.mark.parametrize('stage',['predict','review_response','close'])
def test_failure_closes_without_success_output(cli,capsys,stage):
    module,model,calls,path,payload=cli
    original=model.close
    def fail(*a,**kw):
        if stage=='close': original()
        raise RuntimeError('synthetic failure')
    setattr(model,stage,fail)
    with pytest.raises(SystemExit) as e: module.main(args(path)+['--draft','Reply','--context',''])
    captured=capsys.readouterr()
    assert e.value.code==2 and not captured.out and 'synthetic failure' in captured.err
    assert calls[-1]==('close',)

@pytest.mark.parametrize('neural',[False,True])
def test_legacy_selector_neural_and_response_remain(cli,monkeypatch,capsys,neural):
    module,model,calls,path,payload=cli
    class Selector:
        def __init__(self,path,*,expected_sha256): calls.append(('selector',expected_sha256))
        def predict(self,payload,**kwargs): calls.append(('legacy',kwargs));return {'probabilities':[1],'qualified':False}
        def review_response(self,draft,language,context): calls.append(('sparse',context)); return {'legacy':True}
    legacy=types.ModuleType('roguard.trajectory_model');legacy.TrajectoryModel=Selector
    monkeypatch.setitem(sys.modules,legacy.__name__,legacy)
    neuralmodule=types.ModuleType('roguard.trajectory_neural')
    neuralmodule.NeuralTrajectoryModel=types.SimpleNamespace(from_profile=lambda s:s)
    monkeypatch.setitem(sys.modules,neuralmodule.__name__,neuralmodule)
    argv=[str(path),'--model','legacy.json','--model-sha256','a'*64,'--draft','Reply']
    module.main(argv+(['--neural'] if neural else []))
    assert json.loads(capsys.readouterr().out)=={'qualified':False,'response_review':{'legacy':True}}
    assert ('legacy',{'calibration':None,'plan':False}) in calls
    assert ('sparse',payload['turns'][-1]['text']) in calls

def test_core_cli_import_and_lazy_export_without_numpy(monkeypatch):
    import roguard
    original=__import__
    def guarded(name,*a,**kw):
        if name.split('.')[0] in {'numpy','scipy','mlx','torch'}: raise AssertionError('eager numerical import')
        return original(name,*a,**kw)
    with mock.patch('builtins.__import__',guarded):
        spec=importlib.util.spec_from_file_location('standalone_cli',package_source('trajectory_cli.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        child=types.ModuleType('roguard.v10_bundle');sentinel=object();child.load_v10=sentinel
        monkeypatch.setitem(sys.modules,child.__name__,child)
        assert roguard.__getattr__('load_v10') is sentinel
        assert 'load_v10' in roguard.__all__


def test_bundle_calibration_and_legacy_context_rejection(cli,capsys):
    module,model,calls,path,payload=cli
    calibration=path.with_name('calibration.json');calibration.write_text('{"pipeline_digest":"bound"}')
    module.main(args(path)+['--calibration',str(calibration)])
    assert calls[1]==('predict',payload,{'calibration':{'pipeline_digest':'bound'}})
    calls.clear()
    with pytest.raises(SystemExit):
        module.main([str(path),'--model','legacy','--model-sha256','a'*64,'--draft','Reply','--context','x'])
    assert not calls


@pytest.mark.parametrize('custom_cache',[False,True])
def test_profile_resolves_only_cached_explicit_assets(cli,monkeypatch,custom_cache):
    module,model,calls,path,payload=cli
    installer=types.ModuleType('roguard.model_install')
    installer.profile=lambda name: (b'catalog manifest','catalog digest')
    installer.cache_directory=lambda: Path('default-cache')
    def install(raw,sha,cache,*,download):
        calls.append(('install',raw,sha,str(cache),download))
        return Path('cached/bundle.json'),'b'*64
    installer.install=install
    monkeypatch.setitem(sys.modules,installer.__name__,installer)
    module.main([str(path),'--profile','explicit-profile']+(['--cache','custom-cache'] if custom_cache else []))
    assert calls[0]==('install',b'catalog manifest','catalog digest','custom-cache' if custom_cache else 'default-cache',False)
    assert calls[1]==('load','cached/bundle.json','b'*64)
    assert calls[-1]==('close',)

@pytest.mark.parametrize('flags',[['--profile','x','--bundle-sha256','a'*64],['--profile','x','--neural'],['--profile','x','--draft','Reply'],['--bundle','x','--bundle-sha256','a'*64,'--cache','x']])
def test_profile_conflicts_before_loading(cli,flags):
    module,model,calls,path,payload=cli
    with pytest.raises(SystemExit) as e: module.main([str(path)]+flags)
    assert e.value.code==2 and not calls

def test_profile_missing_cache_fails_without_loader(cli,monkeypatch,capsys):
    module,model,calls,path,payload=cli
    installer=types.ModuleType('roguard.model_install')
    installer.profile=lambda name:(b'raw','sha')
    installer.cache_directory=lambda:Path('cache')
    def install(*args,**kwargs):
        assert kwargs=={'download':False}
        raise ValueError('Model files are missing or changed; run roguard-fetch fetch explicitly')
    installer.install=install
    monkeypatch.setitem(sys.modules,installer.__name__,installer)
    with pytest.raises(SystemExit) as e: module.main([str(path),'--profile','x'])
    assert e.value.code==2 and not calls
    assert 'roguard-fetch fetch explicitly' in capsys.readouterr().err


def test_help_portable_and_python39_syntax(cli,capsys):
    module,model,calls,path,payload=cli
    for name in ('trajectory_cli.py','__init__.py'):
        ast.parse(package_source(name).read_text(),feature_version=(3,9))
    original=__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in {'numpy','scipy','mlx','torch'}:
            raise AssertionError('optional numerical import during help')
        return original(name,*args,**kwargs)
    with mock.patch('builtins.__import__',guarded), pytest.raises(SystemExit) as e:
        module.main(['--help'])
    assert e.value.code==0 and not calls
    output=capsys.readouterr().out
    assert '--profile' in output and '--bundle' in output and '--context' in output
