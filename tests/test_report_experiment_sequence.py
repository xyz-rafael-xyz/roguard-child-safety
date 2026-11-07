import json
import os

import pytest
from scripts.run_report_sequence_v10 import dependency_state


def report(tmp_path, data):
    path=tmp_path/'report.json';path.write_text(json.dumps(data));return path


def test_live_unfinished_dependency_waits(tmp_path):
    assert dependency_state(report(tmp_path,{'phase':'running','execution_complete':False}),os.getpid()) is None


def test_scientific_rejection_does_not_mean_execution_failure(tmp_path):
    path=report(tmp_path,{'phase':'finished','execution_complete':True,'decision':'REJECT',
                          'completed_steps':[{'exit_code':0}]})
    assert len(dependency_state(path,os.getpid()))==64


@pytest.mark.parametrize('phase',['step_failed','dependency_failed','cancelled'])
def test_failed_dependency_never_launches_work(tmp_path,phase):
    with pytest.raises(ValueError,match='failed'):
        dependency_state(report(tmp_path,{'phase':phase}),os.getpid())


def test_false_completion_rejected(tmp_path):
    with pytest.raises(ValueError,match='failed steps'):
        dependency_state(report(tmp_path,{'phase':'finished','execution_complete':True,
                                         'completed_steps':[{'exit_code':1}]}),os.getpid())


def test_dead_dependency_cannot_be_treated_as_completed(tmp_path,monkeypatch):
    def missing(*args):raise ProcessLookupError()
    monkeypatch.setattr(os,'kill',missing)
    with pytest.raises(ValueError,match='without a complete'):
        dependency_state(report(tmp_path,{'phase':'running'}),123)
