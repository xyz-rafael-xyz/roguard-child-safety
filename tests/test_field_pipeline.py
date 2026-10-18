from types import SimpleNamespace
import pytest

from roguard.field_pipeline import FieldReviewPipeline, _local_path


class Model:
    def __init__(self, identity): self.identity, self.calls = identity, []
    def predict_many(self, payloads, *, focuses):
        self.calls.append((payloads, focuses))
        return [{'identity':self.identity} for _ in payloads]


def pipeline():
    fusion=SimpleNamespace(identity='fusion',_spec=dict(language='ro',native_heads_sha256='a',pretrained_heads_sha256='b'),
        predict=lambda payload,a,b,focus:dict(models=[a['identity'],b['identity']],focus=focus))
    return FieldReviewPipeline(fusion,Model('a'),Model('b'))


def test_pipeline_preserves_explicit_focus():
    model=pipeline()
    payload=dict(language='ro',turns=[dict(role='minor',text='Exemplu.'),dict(role='minor',text='Alt exemplu.')])
    assert model.predict(payload,focus=0)==dict(models=['a','b'],focus=0)
    assert model.native.calls[0][1]==[0] and model.pretrained.calls[0][1]==[0]


def test_wrong_language_rejected_before_neural_loading():
    model=pipeline()
    with pytest.raises(ValueError):model.predict(dict(language='uk',turns=[dict(role='minor',text='Приклад.')]))
    assert not model.native.calls and not model.pretrained.calls


@pytest.mark.parametrize('value',['../outside','/outside',''])
def test_bundle_path_escape_rejected(tmp_path,value):
    with pytest.raises(ValueError):_local_path(tmp_path,value)


def test_symlink_escape_rejected(tmp_path):
    (tmp_path/'link').symlink_to(tmp_path.parent,target_is_directory=True)
    with pytest.raises(ValueError):_local_path(tmp_path,'link/file')


def test_local_asset_path_accepted(tmp_path):
    assert _local_path(tmp_path,'base/config.json')==tmp_path/'base/config.json'
