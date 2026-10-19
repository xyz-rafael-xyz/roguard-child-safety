import hashlib
import json
import pytest
from training.resume_qwen_audit_v9 import POLICY, read_checkpoint
from training.evaluate_qwen_adapter_v9 import POLICY as ORIGINAL_POLICY


def checkpoint(tmp_path, changes=None):
    rows = [dict(case_id=str(i), language='uk') for i in range(3)]
    values = [dict(rows[i], gold=0, score=.25, predicted=0, valid_label_token=True) for i in range(2)]
    if changes:
        values[0].update(changes)
    progress = tmp_path / 'progress.jsonl'
    progress.write_text(''.join(json.dumps(r) + '\n' for r in values))
    record = dict(progress_sha256=hashlib.sha256(progress.read_bytes()).hexdigest(),
        inputs_sha256='a', adapter_sha256='b', original_evaluator_sha256='c',
        total_rows=3, completed_rows=2)
    path = tmp_path / 'pause.json'
    path.write_text(json.dumps(record))
    args = dict(data_sha256='a', adapter_sha256='b', evaluator_sha256='c')
    return rows, progress, path, args


def test_resume_preserves_the_original_policy_and_completed_prefix(tmp_path):
    assert POLICY == ORIGINAL_POLICY
    rows, progress, record, args = checkpoint(tmp_path)
    prior, _ = read_checkpoint(rows, [0, 0, 0], progress, record, **args)
    assert [r['case_id'] for r in prior] == ['0', '1']


@pytest.mark.parametrize('changes', [
    dict(score=float('nan')), dict(score=True), dict(score=1.1), dict(gold=True),
    dict(gold=1), dict(valid_label_token=1), dict(predicted=True), dict(predicted=1),
    dict(case_id='other'), dict(language='ro'),
])
def test_invalid_or_rebound_progress_is_rejected(tmp_path, changes):
    rows, progress, record, args = checkpoint(tmp_path, changes)
    with pytest.raises(ValueError):
        read_checkpoint(rows, [0, 0, 0], progress, record, **args)


@pytest.mark.parametrize('key', ['data_sha256', 'adapter_sha256', 'evaluator_sha256'])
def test_changed_candidate_or_population_cannot_resume(tmp_path, key):
    rows, progress, record, args = checkpoint(tmp_path)
    args[key] = 'changed'
    with pytest.raises(ValueError, match='identity'):
        read_checkpoint(rows, [0, 0, 0], progress, record, **args)


def test_partial_write_or_progress_tampering_cannot_resume(tmp_path):
    rows, progress, record, args = checkpoint(tmp_path)
    progress.write_bytes(progress.read_bytes()[:-1])
    with pytest.raises(ValueError, match='identity'):
        read_checkpoint(rows, [0, 0, 0], progress, record, **args)
