"""Reproducible v5 software/reference evaluation; no language-model accuracy claim."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
from roguard.flow import FlowLedger
from roguard.lab import run_suite, policy_pairs
from roguard.lab_scenarios import generate_suite
from roguard.interaction_lab import ReferenceTarget, run_scenario, minimize_failure
from roguard.consistency import evaluate_consistency


def transition_oracle():
    # Independent explicit automaton, with expiry tracked as a separate variable.
    table = {'absent': {'grant':'active','revoke':'absent','pause':'absent','resume':'absent'},
             'active': {'grant':'active','revoke':'absent','pause':'paused','resume':'active'},
             'paused': {'grant':'active','revoke':'absent','pause':'paused','resume':'active'}}
    checked = 0; failures = []
    for size in range(1, 6):
        for actions in itertools.product(('grant','revoke','pause','resume'), repeat=size):
            state = 'absent'; expiry = None
            for i, action in enumerate(actions):
                state = table[state][action]
                if action == 'grant': expiry = 3 if i % 2 else None
                elif action == 'revoke': expiry = None
            for at in (0, 2, 3, 7):
                with FlowLedger() as ledger:
                    ledger.add_source('source','Fictional.'); ledger.derive('summary','Derived.',['source'])
                    for i, action in enumerate(actions):
                        ledger.permission('source','task','recipient','use',action,
                                          expires_at=3 if action=='grant' and i%2 else None)
                    ledger.advance(at)
                    actual = ledger.check(['summary'],'task','recipient','use')['decision']
                    expected = 'ALLOW' if state=='active' and (expiry is None or at<expiry) else 'BLOCK'
                    checked += 1
                    if actual != expected: failures.append({'actions':actions,'at':at,'expected':expected,'actual':actual})
    return {'checks':checked,'failures':failures,'bounded_sequence_length':5,
            'independent_table_oracle':True,'unbounded_correctness_proven':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261025);parser.add_argument('--repetitions',type=int,default=100)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve existing evaluation records')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Commit implementation before final evaluation')
    scenarios=generate_suite(args.seed,args.repetitions);started=time.perf_counter()
    reports={};pairs={}
    for mode,enforce in [('permissive',False),('permissive',True),('overblocking',True),('guarded',True),('stale',True)]:
        name=mode+('_enforced' if enforce else '_unrestricted')
        factory=lambda:ReferenceTarget(mode)
        reports[name]=run_suite(scenarios,factory,enforce=enforce)
        pair_rows=[{**p,'pair_id':language+'_'+p['pair_id']} for language in ('ro','uk') for p in policy_pairs(language,args.seed)]
        pairs[name]=evaluate_consistency(pair_rows,lambda s:run_scenario(s,factory,enforce=enforce)['requests'][-1]['outcome'])
    reduction=minimize_failure(scenarios[0],lambda:ReferenceTarget('stale'))
    # The reduced scenario is reproducible from the generator; retain metadata only here.
    reduction.pop('scenario',None);reduction['text_in_report']=False
    reduction['contains_synthetic_scenario_text']=False
    code_paths=list(Path('src/roguard').glob('*.py'))+[Path(__file__)]
    record={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'seed':args.seed,'repetitions_per_family_language':args.repetitions,
            'scenario_rows':len(scenarios),'base_policy_families':12,'languages':['ro','uk'],
            'scenario_sha256':hashlib.sha256(json.dumps(scenarios,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths if not p.is_absolute()},
            'reference_targets':reports,'paired_reference_checks':pairs,'transition_oracle':transition_oracle(),
            'failure_reduction':reduction,'seconds':time.perf_counter()-started,
            'synthetic_only':True,'independent_language_validation':False,'language_model_accuracy_established':False}
    args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'scenarios':len(scenarios),'transition_checks':record['transition_oracle']['checks'],
                      'transition_failures':len(record['transition_oracle']['failures']),'seconds':record['seconds']}))
    if record['transition_oracle']['failures'] or reports['guarded_enforced']['failures']:
        raise SystemExit('Reference correctness gate failed; results preserved')

if __name__=='__main__':main()
