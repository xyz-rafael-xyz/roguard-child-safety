import json
import sys
import pytest
from roguard import review_cli


def test_one_message_cli_uses_only_cached_profile(monkeypatch,capsys,tmp_path):
    calls=[]
    monkeypatch.setattr(sys,'argv',['roguard-review','--text','Вітаю.','--language','uk'])
    monkeypatch.setattr(review_cli,'profile',lambda name:(b'profile','digest'))
    def install(raw,digest,cache,*,download):
        assert download is False
        import hashlib
        raw=b'{"format":"roguard-entailment-review-bundle-v1"}'
        path=tmp_path/'bundle.json';path.write_bytes(raw)
        return path,hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(review_cli,'install',install)
    class Model:
        def predict(self,payload,*,focus):
            calls.append((payload,focus));return dict(predicted=0,permits_notification=False)
    monkeypatch.setattr('roguard.entailment_bundle.load_bundle',lambda path,expected_sha256:Model())
    review_cli.main()
    assert calls==[(dict(language='uk',turns=[dict(role='minor',text='Вітаю.')]),None)]
    assert json.loads(capsys.readouterr().out)['permits_notification'] is False


@pytest.mark.parametrize('args',[['--text','hello'],['--text','hello','--language','uk','input.json'],
                                 ['--text','Вітаю.','--language','uk','--focus','3']])
def test_invalid_input_is_rejected_before_model_loading(monkeypatch,args):
    monkeypatch.setattr(sys,'argv',['roguard-review']+args)
    monkeypatch.setattr(review_cli,'profile',lambda name:pytest.fail('Loaded invalid input'))
    with pytest.raises(SystemExit) as error:review_cli.main()
    assert error.value.code==2


def test_portable_profile_dependency_error_names_semantic_extra(monkeypatch,capsys,tmp_path):
    import hashlib
    monkeypatch.setattr(sys,'argv',['roguard-review','--text','Bună.','--language','ro'])
    monkeypatch.setattr(review_cli,'profile',lambda name:(b'profile','digest'))
    raw=b'{"format":"roguard-entailment-review-bundle-v1"}'
    path=tmp_path/'bundle.json';path.write_bytes(raw)
    monkeypatch.setattr(review_cli,'install',lambda *a,**k:(path,hashlib.sha256(raw).hexdigest()))
    def missing(*a,**k):raise ImportError('torch is missing')
    monkeypatch.setattr('roguard.entailment_bundle.load_bundle',missing)
    with pytest.raises(SystemExit) as error:review_cli.main()
    assert error.value.code==2
    message=capsys.readouterr().err
    assert 'roguard-child-safety[semantic]' in message
    assert '[fusion]' not in message
