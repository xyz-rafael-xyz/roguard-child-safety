import json

import pytest

from scripts.verify_mixed_response_completion_v10 import split_completed_log


def test_terminal_report_is_verified_without_discarding_update_events():
    prefix = b'{"starting":"train-mixed-large-response-attention65"}\n{"step":25}\n'
    report = {'complete': True, 'selected': {'epoch': 1}}
    raw = prefix+(json.dumps(report, indent=2)+'\n').encode()
    assert split_completed_log(raw, report) == prefix


@pytest.mark.parametrize('fault', ['absent', 'partial', 'changed', 'extra', 'incomplete'])
def test_corrupt_or_incomplete_terminal_report_is_rejected(fault):
    report = {'complete': True, 'selected': {'epoch': 1}}
    prefix = b'{"step":25}\n'
    terminal = (json.dumps(report, indent=2)+'\n').encode()
    raw = prefix+terminal
    if fault == 'absent': raw = prefix
    elif fault == 'partial': raw = raw[:-2]
    elif fault == 'changed': raw = raw.replace(b'"epoch": 1', b'"epoch": 2')
    elif fault == 'extra': raw += terminal
    elif fault == 'incomplete':
        report['complete'] = False
        raw = prefix+(json.dumps(report, indent=2)+'\n').encode()
    with pytest.raises(ValueError):
        split_completed_log(raw, report)
