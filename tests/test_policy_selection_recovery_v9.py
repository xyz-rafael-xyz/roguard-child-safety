import hashlib
import json
from pathlib import Path

from training import select_policy_interpreter_v9 as selection


def test_selection_creates_parent_before_decoder_and_preserves_training_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'training').mkdir()
    (tmp_path/'training/interpreter_validation_v9.py').write_bytes(
        Path(selection.__file__).with_name('interpreter_validation_v9.py').read_bytes())
    root = Path('research/v9-stabilization'); study = root/'interpreter-policy-27'
    parent = root/'interpreter-14'
    (parent/'adapted').mkdir(parents=True); study.mkdir()
    (root/'work-window.json').write_text(json.dumps(dict(deadline_utc='2099-01-01T00:00:00Z')))
    for folder in (study, parent): (folder/'valid.jsonl').write_text('{}\n')
    (parent/'adapted/predictions.jsonl').write_text('{}\n')
    checkpoints = []
    for index, raw in ((0, b'parent'), (255, b'trained')):
        folder = study/'validation-checkpoints'/str(index); folder.mkdir(parents=True)
        (folder/'adapters.safetensors').write_bytes(raw)
        (folder/'adapter_config.json').write_text('{}')
        checkpoints.append(dict(iteration=index, path=str(folder), sha256=hashlib.sha256(raw).hexdigest(), val_loss=.5))
    (study/'config.json').write_text(json.dumps(dict(initial_adapter_sha256=checkpoints[0]['sha256'])))
    (study/'training-metrics.json').write_text(json.dumps(dict(validation_checkpoints=checkpoints)))
    (study/'selected-adapter').mkdir()
    (study/'selected-adapter/adapters.safetensors').write_bytes(b'parent')
    (study/'selected-adapter/adapter_config.json').write_text('{}')
    monkeypatch.setattr(selection.tempfile, 'gettempdir', lambda: str(tmp_path))
    (tmp_path/'roguard-v9-window').mkdir()
    monkeypatch.setattr(selection.sys, 'argv', ['select', '--base', 'unused'])
    def decode(command, **kwargs):
        output = Path(command[command.index('--output')+1])
        assert output.parent.is_dir()
        output.mkdir(exist_ok=False)
        (output/'predictions.jsonl').write_text('{"better":true}\n')
    monkeypatch.setattr(selection.subprocess, 'run', decode)
    monkeypatch.setattr(selection, 'summarize', lambda rows, predictions, loss: dict(
        rank=[0 if predictions[0].get('better') else 1], unweighted_validation_loss=loss))
    selection.main()
    assert (study/'selected-adapter/adapters.safetensors').read_bytes() == b'trained'
    assert (study/'loss-selected-adapter/adapters.safetensors').read_bytes() == b'parent'
    assert json.loads((study/'validation-selection.json').read_text())['selected']['iteration'] == 255
