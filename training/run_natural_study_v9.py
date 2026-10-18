"""Run the frozen natural-06 protocol serially and audit each completed model.

This runner never publishes a release or treats regression passes as fresh
evaluation. It can resume an already completed candidate only after validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from training.check_review_candidate_v8 import inspect
from training.compare_review_candidates_v9 import compare
from training.mix_review_v9 import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--wait-for-pid', type=int)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    if args.wait_for_pid:
        while True:
            try:
                os.kill(args.wait_for_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(10)
    frozen = json.loads((root/'natural-06-v2-source-freeze.json').read_text())
    for path, digest in frozen.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen training source changed: '+path)
    summary = {'stable_release_ready':False,'reserved_probes_used':False,'candidates':{}}
    for language, original in (('ro','review-09-ro'),('uk','review-08-uk')):
        output = root/('natural-06-'+language)
        baseline = Path('research/v8-stabilization')/original
        if output.exists():
            record = json.loads((output/'development.json').read_text())
            if record.get('artifact_sha256') != hashlib.sha256((output/'review.json.gz').read_bytes()).hexdigest():
                raise ValueError('Incomplete existing candidate; inspect it before retrying')
        else:
            cmd = [sys.executable,'-m','training.adapt_natural_v9','--base',args.base,
                '--device',args.device,'--output',str(output),'--initial-review',str(baseline/'review.json.gz'),
                '--train-language',language,'--natural-contrasts','--replay-rows','8000',
                '--structured-rows','0','--conversational-rows','0','--epochs','2',
                '--learning-rate','0.00001','--label-smoothing','0.02']
            print('START',language,flush=True)
            with (root/('natural-06-'+language+'.log')).open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        reports = {'promotion-check':inspect(output,language),
                   'comparison':compare(baseline,output,language),
                   'mixture':evaluate(baseline,output,language)}
        for name,report in reports.items():
            (output/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        summary['candidates'][language] = reports['promotion-check']
        (root/'natural-06-run-status.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('AUDITED',language,json.dumps(reports['promotion-check']['failures']),flush=True)


if __name__=='__main__':
    main()
