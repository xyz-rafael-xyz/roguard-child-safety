"""Freeze the completed reference evidence before either matched training arm."""
import argparse
import json
from pathlib import Path

from roguard.trajectory_data import file_sha
from training.matched_role_attention_inputs_v10 import (
    AUDIT_PLAN, AUDIT_PROOF, AUDIT_QUEUE, AUDIT_SEQUENCE, OPTIMIZATION, POPULATIONS,
    completed_inputs, read)


SOURCES = (
    'training/matched_role_diversity_v10.py',
    'training/matched_role_attention_inputs_v10.py',
    'training/train_matched_role_attention_v10.py',
    'training/register_matched_role_attention_v10.py',
    'training/large_review_attention_v10.py',
    'training/coupled_policy_loss_v10.py',
    'training/dropout_readout_v10.py',
    'training/runtime_attention_continuation_v10.py',
    'training/separation_frontier_v9.py',
    'training/balanced_headroom_frontier_v9.py',
    'tests/test_matched_role_diversity_v10.py',
    'tests/test_matched_role_attention_inputs_v10.py',
    'tests/test_matched_role_attention_training_v10.py',
    'tests/test_matched_role_attention_registration_v10.py',
)
OUTPUTS = {
    'control': {'study': 'research/v10-evidence/matched-role-attention-control-105',
                'private': 'matched-role-attention-control-105'},
    'treatment': {'study': 'research/v10-evidence/matched-role-attention-treatment-106',
                  'private': 'matched-role-attention-treatment-106'},
}


def compose(bindings, public, private):
    if bindings['populations'] != POPULATIONS:
        raise ValueError('Completed matched population differs')
    for lang, counts in POPULATIONS.items():
        expected = dict(updates_per_epoch=(counts['train']+7)//8,
            total_updates=3*((counts['train']+7)//8), total_presentations=3*counts['train'])
        if bindings['schedules'][lang] != expected:
            raise ValueError('Completed matched schedule differs')
        matching = bindings['matching'][lang]
        if (matching['base_train_rows'] != counts['base_train']
                or matching['added_train_slots'] != counts['added_train']
                or matching['train_rows_per_arm'] != counts['train']
                or matching['development_rows'] != counts['dev']):
            raise ValueError('Completed matching counts differ')
    return dict(format='roguard-matched-role-attention-plan-1',
        source_sha256=public['training/train_matched_role_attention_v10.py'],
        registration_source_sha256=public['training/register_matched_role_attention_v10.py'],
        optimization=OPTIMIZATION, populations=POPULATIONS, input_bindings=bindings,
        initialization='selected95', public=public, private=private, outputs=OUTPUTS,
        arms=['control', 'treatment'], train_rows_per_arm=32097, development_rows=8837,
        total_updates_both_arms=24078, total_presentations_both_arms=192582,
        training_admitted=True, held_out_test_opened=False, qualified=False,
        control='Repeat original TRAIN inputs, matched per occurrence by language, all six fields and binary label; '
                'cycle deterministic hash-ordered pools. Repeats are occurrence slots, not distinct examples.',
        treatment='Add the 3168 unchanged revised100 synthetic TRAIN inputs. Original TRAIN and all DEV remain unchanged.',
        selection='Retain initialization and every epoch. Use the existing DEV binary/joint/CE/margin ranking per arm and '
                  'language, then independently audit paired outcomes. Any binary/joint error or v9 introduction rejects '
                  'the engineering candidate. DEV success alone is insufficient for release.',
        scope='A controlled synthetic training experiment motivated by exposed development diagnostics. No independent '
              'language/gold validation or held-out accuracy claim. Prospective fitting registration follows actual '
              'completed reference102 and audit103, before optimizer updates in either arm.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private_root', 'plan'):
        parser.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = parser.parse_args()
    if a.plan.exists():
        raise ValueError('Refusing to overwrite matched training registration')
    for destination in OUTPUTS.values():
        if (a.root/destination['study']).exists() or (a.private_root/destination['private']).exists():
            raise ValueError('Matched training arm output already exists')
    _, bindings = completed_inputs(a.root, a.private_root)
    parent = read(a.root/AUDIT_PLAN)
    prior = read(a.root/'research/v10-evidence/protocol-amendment-74.json')
    public = dict(prior['public'], **parent['public'])
    private = dict(prior['private'], **parent['private'])
    for name in SOURCES+(AUDIT_PLAN, AUDIT_SEQUENCE,
            'research/v10-evidence/role-diversity-runtime-audit-103/queue.json',
            'research/v10-evidence/role-diversity-runtime-102/report.json',
            'research/v10-evidence/role-diversity-runtime-102/protocol.json'):
        public[name] = file_sha(a.root/name)
    for name in (AUDIT_PROOF, AUDIT_QUEUE):
        private[name] = file_sha(a.private_root/name)
    for name, checksum in bindings['reference102_artifacts'].items():
        private['role-diversity-runtime-102/'+name] = checksum
    plan = compose(bindings, public, private)
    from training.role_diversity_runtime_v10 import check_bindings
    check_bindings(a.root, a.private_root, plan)
    a.plan.parent.mkdir(parents=True, exist_ok=True)
    with a.plan.open('x') as handle:
        json.dump(plan, handle, indent=2); handle.write('\n')
    print(json.dumps(dict(registered=True, plan_sha256=file_sha(a.plan), training_launched=False,
                         train_rows_per_arm=32097, development_rows=8837, qualified=False)))


if __name__ == '__main__':
    main()
