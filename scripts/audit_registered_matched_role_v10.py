"""Invoke the paired auditor only after the complete registered pipeline exits."""
import argparse
from pathlib import Path
import subprocess
import sys

from roguard.trajectory_data import file_sha
from scripts.run_matched_role_pipeline_v10 import MARKERS, validate_prospective
from training.matched_role_attention_inputs_v10 import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private_root', 'prospective', 'output'):
        parser.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    parser.add_argument('--prospective-sha256', required=True)
    parser.add_argument('--producer-pid', type=int, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite paired audit')
    plan = validate_prospective(a)
    record = read(a.root/plan['registration_directory']/'report.json')
    stages = ['register-matched-role104', 'preflight-matched-role-control', 'preflight-matched-role-treatment',
              MARKERS['control'], MARKERS['treatment']]
    if (record.get('complete') is not True or record.get('phase') != 'finished'
            or record.get('audit_frozen_before_training') is not True
            or record.get('held_out_test_opened') is not False or record.get('qualified') is not False
            or record.get('prospective_sha256') != a.prospective_sha256
            or record.get('queue_plan_sha256') != file_sha(a.private_root/plan['queue_plan'])
            or [s['name'] for s in record['completed_stages']] != stages
            or any(s['exit_code'] != 0 for s in record['completed_stages'])):
        raise ValueError('Paired pipeline is incomplete or differs from registration')
    audit_path = a.root/plan['audit_plan']
    if (file_sha(audit_path) != record['audit_plan_sha256']
            or file_sha(a.root/plan['training_plan']) != record['training_plan_sha256']):
        raise ValueError('Completed paired plan differs from the pre-training snapshot')
    subprocess.run([sys.executable, 'eval/verify_matched_role_attention_v10.py', '--root', str(a.root),
        '--private-root', str(a.private_root), '--audit-plan', str(audit_path),
        '--audit-plan-sha256', record['audit_plan_sha256'], '--producer-pid', str(a.producer_pid),
        '--output', str(a.output)], cwd=a.root, check=True)


if __name__ == '__main__':
    main()
