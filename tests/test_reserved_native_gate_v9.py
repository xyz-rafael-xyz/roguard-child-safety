import hashlib
import json

import pytest
from training.verify_reserved_native_v9 import checked_report


def evidence(tmp_path, counts):
    split = dict(parity_pass=True, accuracy_gates_pass=True, abstentions=0,
                 decision_disagreements=0, max_score_error=0.0, counts=counts)
    report = dict(format='roguard-installed-native-replay-v1', complete=True,
                  parity_pass=True, accuracy_gates_pass=True, artifact_sha256='a'*64,
                  splits=dict(dev=split, cal=split))
    path = tmp_path/'report.json'
    path.write_text(json.dumps(report))
    return dict(runtime_report=str(path), runtime_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                artifact_sha256='a'*64)


def test_pass_flags_do_not_override_failed_counts(tmp_path):
    failing = dict(positive=100, negative=100, missed=2, false_reviews=0)
    entry = evidence(tmp_path, dict(formal=failing, conversational=failing))
    with pytest.raises(ValueError, match='counts failed'):
        checked_report(entry)


def test_missing_population_cannot_pass(tmp_path):
    entry = evidence(tmp_path, {})
    with pytest.raises(ValueError, match='strata are missing'):
        checked_report(entry)
