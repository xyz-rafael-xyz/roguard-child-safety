import json

import pytest
from training.coordinator_status_v9 import supervise


def test_setup_exception_records_terminal_failure_without_private_error_text(tmp_path):
    def fail():raise RuntimeError('sensitive local configuration')
    with pytest.raises(RuntimeError):supervise(fail,tmp_path)
    raw=(tmp_path/'run-status.json').read_text();status=json.loads(raw)
    assert status['complete'] and status['finished_utc']
    assert status['jobs'][-1]['exit_code']==1
    assert status['jobs'][-1]['error_type']=='RuntimeError'
    assert 'sensitive' not in raw and not status['stable_release_ready']


def test_completed_job_history_is_preserved_on_later_setup_failure(tmp_path):
    (tmp_path/'run-status.json').write_text(json.dumps(dict(jobs=[dict(name='train',exit_code=0)])))
    def fail():raise ValueError('missing checkpoint')
    with pytest.raises(ValueError):supervise(fail,tmp_path)
    status=json.loads((tmp_path/'run-status.json').read_text())
    assert [j['name'] for j in status['jobs']]==['train','coordinator']
