import json

import pytest

from eval.verify_large_review_attention_v10 import audit_training_log


def fixture():
    report = {'decision': 'REJECT', 'languages': {}}
    schedules = {}; events = []; seconds = 0.
    for lang, n in (('ro', 73), ('uk', 121)):
        per = (n+7)//8
        schedules[lang] = {'updates_per_epoch': per, 'total_updates': 3*per}
        checkpoints = [{'epoch': e, 'binary_errors': 4 if e == 0 else 1,
                        'joint_argmax_errors': 8, 'introduced_v9': 1,
                        'selection_rank': [4 if e == 0 else 1, 8, .5, 0.]} for e in range(4)]
        report['languages'][lang] = {'training_rows': n, 'checkpoints': checkpoints}
        for epoch in range(4):
            if epoch:
                for step in range((epoch-1)*per+1, epoch*per+1):
                    if step % 25 and step != epoch*per:
                        continue
                    seconds += 1.
                    events.append({'language': lang, 'epoch': epoch, 'step': step,
                                   'total_steps': 3*per, 'presentations': (epoch-1)*n+min((step-(epoch-1)*per)*8, n),
                                   'loss': .1, 'seconds': seconds})
            seconds += 1.
            events.append({'language': lang, 'epoch': epoch, 'binary_errors': checkpoints[epoch]['binary_errors'],
                           'joint_argmax_errors': 8, 'introduced_v9': 1,
                           'selected_epoch': 0 if epoch == 0 else 1, 'seconds': seconds})
    events.append({'complete': True, 'decision': 'REJECT', 'seconds': seconds+1})
    return report, schedules, events


def raw(events):
    return b'{\n  "earlier_verification": "PASS"\n}\n'+b'{"starting": "train-large-review-attention73"}\n'+b''.join(
        (json.dumps(e)+'\n').encode() for e in events)


def test_complete_update_history_includes_partial_batches_and_earliest_tie():
    report, schedules, events = fixture()
    result = audit_training_log(raw(events), report, schedules)
    assert result['training_events'] == len(events)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'counter', 'epoch', 'selection', 'time', 'loss', 'terminal'])
def test_incomplete_or_changed_training_history_is_rejected(fault):
    report, schedules, events = fixture()
    index = next(i for i, e in enumerate(events) if 'step' in e)
    if fault == 'missing': events.pop(index)
    elif fault == 'duplicate': events.insert(index, dict(events[index]))
    elif fault == 'counter': events[index]['presentations'] += 1
    elif fault == 'epoch': events[index]['epoch'] += 1
    elif fault == 'selection': events[index+1]['selected_epoch'] = 2
    elif fault == 'time': events[index]['seconds'] = 0
    elif fault == 'loss': events[index]['loss'] = float('nan')
    elif fault == 'terminal': events[-1]['complete'] = False
    with pytest.raises(ValueError): audit_training_log(raw(events), report, schedules)
