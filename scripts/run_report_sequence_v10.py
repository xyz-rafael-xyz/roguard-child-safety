"""Serialize registered work after an earlier sequence's terminal report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.run_large_data_sequence_v10 import check_sources, sha


def dependency_state(path, pid):
    """A completed execution is distinct from the model's qualification result."""
    report=json.loads(Path(path).read_text())
    if report.get('execution_complete') is True and report.get('phase')=='finished':
        if any(step.get('exit_code')!=0 for step in report.get('completed_steps',[])):
            raise ValueError('Dependency claims completion with failed steps')
        return sha(path)
    if report.get('phase') in ('step_failed','dependency_failed','cancelled'):
        raise ValueError('Dependency execution failed; queued encoder work will not start')
    try:os.kill(pid,0)
    except ProcessLookupError:raise ValueError('Dependency process ended without a complete execution report') from None
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--study',required=True,type=Path)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());check_sources(plan)
    if args.study.exists():raise ValueError('Refusing to overwrite sequence')
    args.study.mkdir(parents=True)
    import fcntl
    lock=Path(plan['writer_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record={'format':'roguard-report-experiment-sequence-1','plan_sha256':sha(args.plan),
            'source_sha256':sha(__file__),'source_hashes':plan['source_hashes'],
            'execution_complete':False,'completed_steps':[],'phase':'waiting_for_encoder',
            'stable_release_ready':False,'promotion_performed':False}
    def persist():
        temp=args.study/'report.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(args.study/'report.json')
    persist()
    try:
        while True:
            value=dependency_state(plan['dependency_report'],plan['dependency_pid'])
            if value is not None:break
            time.sleep(5)
        record['dependency_report_sha256']=value
    except Exception:
        record['phase']='dependency_failed';persist();raise
    for step in plan['steps']:
        try:check_sources(plan)
        except Exception:
            record['phase']='dependency_failed';persist();raise
        record.update(phase='running',active_step=step['name']);persist()
        print(json.dumps({'starting':step['name']}),flush=True);started=time.monotonic()
        env=dict(os.environ,**plan.get('env',{}),**step.get('env',{}))
        result=subprocess.run(step['command'],cwd=step.get('cwd',plan['cwd']),env=env,check=False)
        record['completed_steps'].append({'name':step['name'],'exit_code':result.returncode,'seconds':time.monotonic()-started})
        if result.returncode:
            record['phase']='step_failed';persist();raise SystemExit(result.returncode)
        persist()
    record.update(phase='finished',execution_complete=True,active_step=None);persist()


if __name__=='__main__':main()
