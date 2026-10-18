from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from training import run_native_runtime_v9 as runner


@pytest.mark.parametrize('failed_phase', ['build', 'install'])
def test_packaging_failure_records_terminal_status_without_starting_model_replay(tmp_path, monkeypatch, failed_phase):
    monkeypatch.chdir(tmp_path)
    research = tmp_path/'research/v9-stabilization'
    study = research/'installed-native-29'
    study.mkdir(parents=True)
    (study/'source-freeze.json').write_text('{}')
    now = datetime.now(timezone.utc)
    (research/'work-window.json').write_text(json.dumps({'deadline_utc': (now+timedelta(hours=1)).isoformat()}))
    prior = research/'interpreter-full-21'
    prior.mkdir()
    (prior/'run-status.json').write_text(json.dumps({'jobs': [
        {'exit_code': 0, 'finished_utc': (now-timedelta(minutes=1)).isoformat()}]*2}))
    working = tmp_path/'work'; working.mkdir()
    monkeypatch.setattr(runner.tempfile, 'mkdtemp', lambda **kwargs: str(working))
    monkeypatch.setattr(runner.tempfile, 'gettempdir', lambda: str(tmp_path))
    calls = []
    def execute(command, **kwargs):
        phase = 'build' if 'build' in command else 'install'
        calls.append(phase)
        assert command[0] == 'builder'
        if phase == 'build' and failed_phase == 'install':
            wheels = Path(command[command.index('--outdir')+1])
            wheels.mkdir()
            (wheels/'fixture.whl').write_bytes(b'only a failure-path test fixture')
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    monkeypatch.setattr('sys.argv', ['runner', '--builder-python', 'builder', '--torch-python', 'must-not-run',
                                    '--ro-base', 'unused', '--uk-base', 'unused'])
    runner.main()
    status = json.loads((study/'run-status.json').read_text())
    assert 'finished_utc' in status
    assert status['jobs'][-1]['exit_code'] == 7
    assert calls == (['build'] if failed_phase == 'build' else ['build', 'install'])
    assert not (study/'run-status.tmp').exists()
    assert status['stable_release_ready'] is False
