"""Require a successful terminal sequence and wait briefly for its PID to exit."""
import argparse
import json
import os
from pathlib import Path
import time


def require_exit(report, pid, timeout=30.):
    if (report.get('execution_complete') is not True or report.get('phase') != 'finished'
            or not report.get('completed_steps')
            or any(step.get('exit_code') != 0 for step in report['completed_steps'])):
        raise ValueError('Dependency execution is not successfully complete')
    if type(pid) is not int or pid <= 0 or not 0 <= timeout <= 30:
        raise ValueError('Invalid process guard arguments')
    deadline = time.monotonic()+timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            raise ValueError('Completed sequence PID remains live; no encoder may start')
        time.sleep(min(.05, max(0., deadline-time.monotonic())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--pid', type=int, required=True)
    a = p.parse_args()
    require_exit(json.loads(a.report.read_text()), a.pid)
    print(json.dumps({'dependency_execution_complete': True, 'dependency_pid_absent': True, 'pid': a.pid}))


if __name__ == '__main__': main()
