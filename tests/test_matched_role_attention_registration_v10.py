from copy import deepcopy

import pytest

from training.register_matched_role_attention_v10 import compose, OUTPUTS
from training.matched_role_attention_inputs_v10 import POPULATIONS, OPTIMIZATION


def fixture():
    bindings = dict(populations=deepcopy(POPULATIONS), schedules={}, matching={})
    for lang, count in POPULATIONS.items():
        updates = (count['train']+7)//8
        bindings['schedules'][lang] = dict(updates_per_epoch=updates, total_updates=3*updates,
                                           total_presentations=3*count['train'])
        bindings['matching'][lang] = dict(base_train_rows=count['base_train'], added_train_slots=count['added_train'],
                                         train_rows_per_arm=count['train'], development_rows=count['dev'])
    public = {'training/train_matched_role_attention_v10.py': 'a'*64,
              'training/register_matched_role_attention_v10.py': 'b'*64}
    return bindings, public, {}


def test_registration_preserves_matched_schedule_and_no_qualification():
    plan = compose(*fixture())
    assert plan['optimization'] == OPTIMIZATION
    assert plan['outputs'] == OUTPUTS and plan['initialization'] == 'selected95'
    assert plan['total_updates_both_arms'] == 2*sum(s['total_updates'] for s in plan['input_bindings']['schedules'].values())
    assert plan['total_presentations_both_arms'] == 2*sum(s['total_presentations'] for s in plan['input_bindings']['schedules'].values())
    assert plan['training_admitted'] and not plan['held_out_test_opened'] and not plan['qualified']


@pytest.mark.parametrize('fault', ['population', 'updates', 'presentations', 'base', 'added', 'total', 'dev'])
def test_registration_rejects_partial_or_unmatched_population(fault):
    bindings, public, private = fixture()
    if fault == 'population': bindings['populations']['ro']['dev'] -= 1
    elif fault == 'updates': bindings['schedules']['uk']['total_updates'] -= 1
    elif fault == 'presentations': bindings['schedules']['ro']['total_presentations'] -= 1
    else:
        key = {'base': 'base_train_rows', 'added': 'added_train_slots', 'total': 'train_rows_per_arm', 'dev': 'development_rows'}[fault]
        bindings['matching']['ro'][key] -= 1
    with pytest.raises(ValueError):
        compose(bindings, public, private)
