import hashlib
import json
import pytest
from roguard.runtime_requirements import runtime_extra


def write(path, spec):
    raw=json.dumps(spec).encode();path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def bundle(tmp_path, formats):
    entries={}
    for language, kind in zip(('ro','uk'),formats):
        path=tmp_path/(language+'.json');digest=write(path,dict(format=kind))
        entries[language]=dict(path=path.name,sha256=digest,device='cpu')
    spec=dict(format='roguard-language-review-bundle-v1',languages=entries)
    path=tmp_path/'bundle.json'
    return path,write(path,spec),spec


def test_bilingual_native_bundle_uses_portable_runtime(tmp_path):
    path,digest,_=bundle(tmp_path,['roguard-native-review-bundle-v1']*2)
    assert runtime_extra(path,digest)=='semantic'


def test_mixed_composite_requires_fusion_runtime(tmp_path):
    path,digest,_=bundle(tmp_path,['roguard-native-review-bundle-v1','roguard-scope-consensus-bundle-v1'])
    assert runtime_extra(path,digest)=='fusion'


def test_changed_child_is_rejected_before_dependency_advice(tmp_path):
    path,digest,_=bundle(tmp_path,['roguard-native-review-bundle-v1']*2)
    (tmp_path/'uk.json').write_text('{}')
    with pytest.raises(ValueError,match='identity'):
        runtime_extra(path,digest)


def test_nested_and_escaping_child_bundles_are_rejected(tmp_path):
    path,digest,spec=bundle(tmp_path,['roguard-language-review-bundle-v1','roguard-native-review-bundle-v1'])
    with pytest.raises(ValueError,match='Nested'):
        runtime_extra(path,digest)
    spec['languages']['ro']['path']='../outside.json'
    with pytest.raises(ValueError):
        runtime_extra(path,write(path,spec))


def test_cli_recommends_portable_extra_for_both_native_languages(tmp_path,monkeypatch,capsys):
    import sys
    from roguard import review_cli
    path,digest,_=bundle(tmp_path,['roguard-native-review-bundle-v1']*2)
    monkeypatch.setattr(sys,'argv',['roguard-review','--language','ro','--text','Bună.'])
    monkeypatch.setattr(review_cli,'profile',lambda name:(b'profile','digest'))
    monkeypatch.setattr(review_cli,'install',lambda *a,**k:(path,digest))
    def missing(*a,**k):raise ImportError('Missing torch')
    monkeypatch.setattr('roguard.entailment_bundle.load_bundle',missing)
    with pytest.raises(SystemExit):review_cli.main()
    assert 'roguard-child-safety[semantic]' in capsys.readouterr().err
