"""A failed model fit must not be mistaken for a completed execution diagnostic."""
import json
from types import SimpleNamespace

import pytest

from training.extract_assertion_continuation_v10 import require_dependency
from roguard.trajectory_data import file_sha


@pytest.fixture
def dependency(tmp_path, monkeypatch):
    study = tmp_path/'research/v10-evidence/execution-layer-diagnostic-84'
    study.mkdir(parents=True)
    source = tmp_path/'eval/verify_execution_layers_v10.py'
    source.parent.mkdir(); source.write_text('test verifier')
    (study/'protocol.json').write_text('{}')
    report = dict(complete=True, optimizer_updates=0, decision='DIAGNOSTIC_ONLY_NO_MODEL_QUALIFICATION')
    (study/'report.json').write_text(json.dumps(report))
    proof = dict(verification='PASS', encoder_loaded=False, source_sha256=file_sha(source),
        report_sha256=file_sha(study/'report.json'), protocol_sha256=file_sha(study/'protocol.json'),
        languages={'ro': {'cases': 9}, 'uk': {'cases': 5}})
    (study/'verification.json').write_text(json.dumps(proof))
    def absent(pid, signal):
        raise ProcessLookupError
    monkeypatch.setattr('training.extract_assertion_continuation_v10.os.kill', absent)
    return SimpleNamespace(root=tmp_path), {'dependency_pid': 123}, study, report, proof


def test_verified_diagnostic_allows_separate_frozen_v9_extraction(dependency):
    args, plan, study, _, _ = dependency
    assert require_dependency(args, plan) == {
        'diagnostic_report_sha256': file_sha(study/'report.json'),
        'verification_sha256': file_sha(study/'verification.json')}


@pytest.mark.parametrize('kind', ['incomplete', 'trained', 'changed_report', 'wrong_inventory', 'failed_proof'])
def test_invalid_diagnostic_cannot_launch_encoder(dependency, kind):
    args, plan, study, report, proof = dependency
    if kind == 'incomplete': report['complete'] = False
    if kind == 'trained': report['optimizer_updates'] = 1
    if kind == 'changed_report': report['changed'] = True
    if kind == 'wrong_inventory': proof['languages']['ro']['cases'] = 8
    if kind == 'failed_proof': proof['verification'] = 'FAIL'
    (study/'report.json').write_text(json.dumps(report))
    (study/'verification.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='not verified'):
        require_dependency(args, plan)


def test_live_encoder_prevents_overlap(dependency, monkeypatch):
    args, plan, _, _, _ = dependency
    monkeypatch.setattr('training.extract_assertion_continuation_v10.os.kill', lambda *args: None)
    with pytest.raises(ValueError, match='still live'):
        require_dependency(args, plan)
