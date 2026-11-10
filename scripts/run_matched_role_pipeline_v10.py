"""Register completed inputs, freeze the audit, and run both matched arms serially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from roguard.trajectory_data import file_sha
from training.matched_role_attention_inputs_v10 import OPTIMIZATION, POPULATIONS, read
from training.register_matched_role_attention_v10 import OUTPUTS
from training.role_diversity_runtime_v10 import check_bindings

MARKERS = {'control': 'train-matched-role-control105', 'treatment': 'train-matched-role-treatment106'}


def validate_prospective(a):
    if file_sha(a.prospective) != a.prospective_sha256:
        raise ValueError('Prospective paired registration differs')
    plan = read(a.prospective)
    if (plan['format'] != 'roguard-matched-role-pipeline-registration-1'
            or plan['optimization'] != OPTIMIZATION or plan['populations'] != POPULATIONS
            or plan['outputs'] != OUTPUTS or plan['markers'] != MARKERS
            or plan['training_performed'] is not False or plan['held_out_test_opened'] is not False
            or plan['qualified'] is not False):
        raise ValueError('Prospective paired scope or design differs')
    check_bindings(a.root, a.private_root, plan)
    return plan


def trainer_command(a, plan, arm, training_sha, *, preflight=False):
    command = [sys.executable, 'training/train_matched_role_attention_v10.py',
        '--root', str(a.root), '--private-root', str(a.private_root), '--base', str(a.base),
        '--study', str(a.root/OUTPUTS[arm]['study']), '--private', str(a.private_root/OUTPUTS[arm]['private']),
        '--plan', str(a.root/plan['training_plan']), '--plan-sha256', training_sha, '--arm', arm]
    if preflight:
        command += ['--preflight-report', str(a.root/plan['registration_directory']/('preflight-'+arm+'.json'))]
    return command


def run(a):
    plan = validate_prospective(a)
    from scripts.wait_completed_sequence_exit_v10 import require_exit
    require_exit(read(a.root/plan['dependency_sequence']), plan['dependency_pid'])
    queue_path = a.private_root/plan['queue_plan']; queue = read(queue_path); queue_sha = file_sha(queue_path)
    relative = str(a.prospective.relative_to(a.root))
    if (queue['source_hashes'].get(relative) != a.prospective_sha256
            or any(queue['source_hashes'].get(name) != sha for name, sha in plan['public'].items())):
        raise ValueError('Producer queue does not bind the prospective sources')
    directory = a.root/plan['registration_directory']; training_path = a.root/plan['training_plan']
    audit_path = a.root/plan['audit_plan']
    if directory.exists() or training_path.exists() or audit_path.exists():
        raise ValueError('Refusing to overwrite paired registration artifacts')
    for value in OUTPUTS.values():
        if (a.root/value['study']).exists() or (a.private_root/value['private']).exists():
            raise ValueError('Refusing to reuse a matched arm output')
    directory.mkdir(parents=True)
    record = dict(format='roguard-matched-role-pipeline-execution-1', complete=False, phase='registering',
        prospective_sha256=a.prospective_sha256, queue_plan_sha256=queue_sha, completed_stages=[],
        held_out_test_opened=False, qualified=False)
    def persist():
        temp = directory/'report.tmp'; temp.write_text(json.dumps(record, indent=2)+'\n'); temp.replace(directory/'report.json')
    def check():
        if validate_prospective(a) != plan or file_sha(queue_path) != queue_sha:
            raise ValueError('Prospective or producer queue changed')
    def stage(name, command):
        check(); record['phase'] = name; persist()
        print(json.dumps({'starting': name}), flush=True)
        started = time.monotonic(); result = subprocess.run(command, cwd=a.root, check=False)
        record['completed_stages'].append(dict(name=name, exit_code=result.returncode, seconds=time.monotonic()-started))
        persist()
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
    persist()
    try:
        stage('register-matched-role104', [sys.executable, 'training/register_matched_role_attention_v10.py',
            '--root', str(a.root), '--private-root', str(a.private_root), '--plan', str(training_path)])
        training_sha = file_sha(training_path); record['training_plan_sha256'] = training_sha
        for arm in ('control', 'treatment'):
            stage('preflight-matched-role-'+arm, trainer_command(a, plan, arm, training_sha, preflight=True))
            proof = read(directory/('preflight-'+arm+'.json'))
            if (proof.get('complete') is not True or proof.get('arm') != arm
                    or proof.get('plan_sha256') != training_sha
                    or any(proof.get(k) is not False for k in
                           ('encoder_loaded', 'training_launched', 'held_out_test_opened', 'qualified'))):
                raise ValueError('Matched arm preflight incomplete or invalid')
        check()
        public = dict(plan['public'], **{relative: a.prospective_sha256, plan['training_plan']: training_sha})
        for arm in ('control', 'treatment'):
            name = plan['registration_directory']+'/preflight-'+arm+'.json'; public[name] = file_sha(a.root/name)
        private = dict(plan['private'], **{plan['queue_plan']: queue_sha})
        audit = dict(format='roguard-matched-role-attention-audit-plan-1', public=public, private=private,
            training_plan=plan['training_plan'], training_plan_sha256=training_sha,
            sequence=plan['sequence'], queue_plan=plan['queue_plan'], log=plan['log'], markers=MARKERS,
            expected_checkpoints=16, checkpoint_dev_rows=70696, reload_rows=17674, initial_new_rows=6336,
            encoder_loaded=False, held_out_test_opened=False, model_qualified=False)
        check_bindings(a.root, a.private_root, audit)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open('x') as handle:
            json.dump(audit, handle, indent=2); handle.write('\n')
        audit_sha = file_sha(audit_path); record['audit_plan_sha256'] = audit_sha
        record['audit_frozen_before_training'] = True; persist()
        from eval.verify_matched_role_attention_v10 import require_complete
        for arm in ('control', 'treatment'):
            if file_sha(training_path) != training_sha or file_sha(audit_path) != audit_sha:
                raise ValueError('Completed-input training or audit plan changed')
            check_bindings(a.root, a.private_root, audit)
            stage(MARKERS[arm], trainer_command(a, plan, arm, training_sha))
            require_complete(read(a.root/OUTPUTS[arm]['study']/'report.json'), arm)
        check(); check_bindings(a.root, a.private_root, audit)
        if file_sha(training_path) != training_sha or file_sha(audit_path) != audit_sha:
            raise ValueError('Registered plans changed during paired fitting')
        record.update(complete=True, phase='finished'); persist()
        # Keep the final trainer completion as the last stdout event. The audit
        # verifies the two strictly separated training log segments.
    except Exception:
        record['phase'] = 'failed'; persist(); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private_root', 'base', 'prospective'):
        parser.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    parser.add_argument('--prospective-sha256', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
