"""Run registered local experiments sequentially after a locked dependency ends."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def check_sources(plan):
    for path,expected in plan['source_hashes'].items():
        if sha(Path(plan['cwd'])/path)!=expected:raise ValueError('Queued source binding changed: '+path)


def completed_dependency(plan):
    path=Path(plan['dependency_report'])
    if not path.is_file() or json.loads(path.read_text()).get('complete') is not True:
        raise ValueError('Dependency did not finish successfully; no queued command may start')
    return sha(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());check_sources(plan)
    if a.study.exists():raise ValueError('Refusing to overwrite sequence')
    a.study.mkdir(parents=True)
    record={'format':'roguard-local-experiment-sequence-1','plan_sha256':sha(a.plan),'source_sha256':sha(__file__),
            'source_hashes':plan['source_hashes'],'execution_complete':False,'completed_steps':[],
            'phase':'waiting_for_encoder','stable_release_ready':False,'promotion_performed':False}
    output=a.study/'report.json'
    def persist():
        temp=a.study/'report.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(output)
    persist()
    import fcntl
    lock=Path(plan['dependency_lock']).open('a')
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(5)
    try:record['dependency_report_sha256']=completed_dependency(plan)
    except Exception as exc:
        record.update(phase='dependency_failed',error=str(exc));persist();raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    import os
    env=dict(os.environ,**plan.get('env',{}))
    for step in plan['steps']:
        check_sources(plan);record.update(phase='running',active_step=step['name']);persist()
        started=time.monotonic()
        print(json.dumps({'starting':step['name']}),flush=True)
        result=subprocess.run(step['command'],cwd=plan['cwd'],env=env,check=False)
        record['completed_steps'].append({'name':step['name'],'exit_code':result.returncode,'seconds':time.monotonic()-started})
        if result.returncode:
            record['phase']='step_failed';persist();raise SystemExit(result.returncode)
        persist()
    record.update(execution_complete=True,phase='finished',active_step=None);persist();print(json.dumps(record),flush=True)


if __name__=='__main__':main()
