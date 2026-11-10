"""Bind the completed-input registration before invoking the independent auditor."""
import argparse
from pathlib import Path
import subprocess
import sys

from training.register_runtime_attention_continuation_v10 import paths, verify_completed_registration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private_root', 'registration', 'output'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--sequence-pid', type=int, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite completion audit')
    checksum = verify_completed_registration(a, require_preflight=True)
    subprocess.run([sys.executable, 'eval/verify_runtime_review_attention_continuation_v10.py',
        '--root', str(a.root), '--private', str(a.private_root), '--amendment', str(paths(a.root)[0]),
        '--amendment-sha256', checksum, '--sequence-pid', str(a.sequence_pid), '--output', str(a.output)],
        cwd=a.root, check=True)


if __name__ == '__main__':
    main()
