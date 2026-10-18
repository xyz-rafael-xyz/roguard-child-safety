import hashlib
import json

import pytest

from training.verify_reserved_discrete_fields_v9 import checked_report


def evidence(tmp_path):
    counts = dict(positive=100, negative=100, missed=1, false_reviews=5, abstentions=0, passes=True)
    split = dict(parity_pass=True, accuracy_gates_pass=True, abstentions=0,
                 decision_changes=0, interpretation_changes=0, maximum_field_error=1e-5,
                 counts={k: dict(counts) for k in ('formal', 'conversational')})
    from roguard.discrete_fields import CONTRACT_DIGEST
    report = dict(contract_digest=CONTRACT_DIGEST, format='roguard-installed-discrete-fields-replay-v1', complete=True,
                  parity_pass=True, accuracy_gates_pass=True, artifact_sha256='heads',
                  native_artifact_sha256='native', selection_sha256='selection',
                  splits={k: json.loads(json.dumps(split)) for k in ('dev', 'cal')})
    entry = dict(runtime_report=str(tmp_path / 'report.json'), artifact_sha256='heads',
                 native_artifact_sha256='native', selection_sha256='selection')
    return report, entry


def write(report, entry):
    from pathlib import Path
    path = Path(entry['runtime_report'])
    path.write_text(json.dumps(report))
    entry['runtime_report_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_gate_boundaries_pass(tmp_path):
    report, entry = evidence(tmp_path)
    write(report, entry)
    assert checked_report(entry) == report


@pytest.mark.parametrize('change', ['recall', 'false_reviews', 'field_parity', 'nan',
                                   'abstention', 'missing_stratum', 'wrong_parent', 'bool_count', 'contract', 'interpretation'])
def test_bad_evidence_cannot_open_reserved_inputs(tmp_path, change):
    report, entry = evidence(tmp_path)
    split = report['splits']['cal']
    group = split['counts']['conversational']
    if change == 'recall': group['missed'] = 2
    elif change == 'false_reviews': group['false_reviews'] = 6
    elif change == 'field_parity': split['maximum_field_error'] = .001
    elif change == 'nan': split['maximum_field_error'] = float('nan')
    elif change == 'abstention': group['abstentions'] = 1
    elif change == 'missing_stratum': del split['counts']['formal']
    elif change == 'wrong_parent': report['native_artifact_sha256'] = 'other'
    elif change == 'bool_count': group['missed'] = True
    elif change == 'contract': report['contract_digest'] = 'other'
    elif change == 'interpretation': split['interpretation_changes'] = 1
    write(report, entry)
    with pytest.raises(ValueError): checked_report(entry)


def test_changed_report_rejected(tmp_path):
    report, entry = evidence(tmp_path)
    write(report, entry)
    entry['runtime_report_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='changed'): checked_report(entry)
