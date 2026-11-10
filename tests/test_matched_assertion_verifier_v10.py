"""Reject altered paired evidence, checkpoint selection, and completed fit logs."""
from copy import deepcopy
import json

import pytest

from eval.verify_matched_assertion_readout_v10 import ARMS, audit_log, paired_counts, verify_selections


def history_fixture():
    history = []
    for arm in ARMS:
        for epoch in range(13):
            entry = {'candidate': arm, 'epoch': epoch, 'selection_rank': [12-epoch, 20, .5, 0.],
                     'checkpoint_sha256': f'{arm}:{epoch}', 'development_logits_sha256': f'logits:{arm}:{epoch}',
                     'completed_training_updates': epoch*213, 'completed_training_presentations': epoch*13614}
            # Only strictly improved overall checkpoints get an export identity.
            if arm == ARMS[0]: entry['adapter_sha256'] = f'export:{epoch}'
            if epoch: entry['training_loss'] = .25
            history.append(entry)
    best = history[12]
    selected = {k: v for k, v in best.items() if k != 'training_loss'}
    arms = {arm: {k: v for k, v in history[i*13+12].items() if k not in ('training_loss', 'adapter_sha256')}
            for i, arm in enumerate(ARMS)}
    return history, selected, arms


def test_equal_arm_rank_preserves_earlier_control_and_optional_fields():
    history, selected, arms = history_fixture()
    best, per_arm = verify_selections(history, selected, arms)
    assert best is history[12]
    assert per_arm[ARMS[1]] is history[25]


@pytest.mark.parametrize('fault', ['missing', 'reordered', 'duplicate', 'late_global_tie',
                                 'late_epoch_tie', 'wrong_arm', 'missing_arm', 'wrong_export', 'wrong_loss'])
def test_checkpoint_selection_corruption_fails(fault):
    history, selected, arms = history_fixture()
    if fault == 'missing': history.pop()
    elif fault == 'reordered': history[0], history[1] = history[1], history[0]
    elif fault == 'duplicate': history[-1] = deepcopy(history[-2])
    elif fault == 'late_global_tie': selected = {k: v for k, v in history[-1].items() if k != 'training_loss'}
    elif fault == 'late_epoch_tie': history[11]['selection_rank'] = list(history[12]['selection_rank'])
    elif fault == 'wrong_arm': arms[ARMS[1]] = dict(arms[ARMS[0]])
    elif fault == 'missing_arm': del arms[ARMS[1]]
    elif fault == 'wrong_export': selected['adapter_sha256'] = 'wrong'
    else: selected['training_loss'] = .25
    with pytest.raises(ValueError): verify_selections(history, selected, arms)


def log_fixture():
    languages = {}
    lines = [json.dumps({'earlier_dependency': 'PASS'}), json.dumps({'starting': 'train-matched-assertion-readout79'})]
    for lang in ('ro', 'uk'):
        history, selected, arms = history_fixture()
        languages[lang] = {'epochs': history, 'selected': selected, 'arm_selected': arms}
        lines.extend(json.dumps({'language': lang, **{k: v for k, v in r.items() if k != 'training_loss'}})
                     for r in history)
    lines.append(json.dumps({lang: r['selected'] for lang, r in languages.items()}, indent=2))
    return ('\n'.join(lines)+'\n').encode(), {'languages': languages}


def test_log_counts_checkpoint_events_without_claiming_minibatch_trace():
    raw, report = log_fixture()
    assert audit_log(raw, report) == {'checkpoint_events': 52, 'per_minibatch_log_available': False}


@pytest.mark.parametrize('fault', ['missing_start', 'duplicate_start', 'missing_checkpoint', 'extra_checkpoint',
                                 'wrong_language', 'wrong_counter', 'reordered', 'wrong_terminal', 'missing_terminal'])
def test_training_log_corruption_fails(fault):
    raw, report = log_fixture(); lines = raw.splitlines()
    if fault == 'missing_start': lines.pop(1)
    elif fault == 'duplicate_start': lines.insert(1, lines[1])
    elif fault == 'missing_checkpoint': lines.pop(2)
    elif fault == 'extra_checkpoint': lines.insert(2, lines[2])
    elif fault == 'wrong_language': lines[2] = lines[2].replace(b'"ro"', b'"uk"')
    elif fault == 'wrong_counter': lines[2] = lines[2].replace(b'"completed_training_updates": 0', b'"completed_training_updates": 1')
    elif fault == 'reordered': lines[2], lines[3] = lines[3], lines[2]
    elif fault == 'wrong_terminal': report['languages']['uk']['selected']['epoch'] = 11
    else: lines = lines[:54]
    with pytest.raises(ValueError): audit_log(b'\n'.join(lines), report)


def test_net_error_reduction_does_not_hide_introductions():
    counts = paired_counts([0, 1, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 1, 0, 0])
    assert counts == {'rows': 5, 'baseline_errors': 3, 'candidate_errors': 2,
                      'fixes': 2, 'introductions': 1, 'both_wrong': 1, 'both_correct': 1}


@pytest.mark.parametrize('gold,before,after', [([], [], []), ([0], [], [0]), ([0], [0], [0, 1]),
                                            ([2], [0], [0]), ([0], [-1], [0]), ([0], [0], [float('nan')])])
def test_invalid_paired_populations_rejected(gold, before, after):
    with pytest.raises(ValueError): paired_counts(gold, before, after)
