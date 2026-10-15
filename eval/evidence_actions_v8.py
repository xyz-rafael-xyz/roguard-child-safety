"""Exhaustive declared-policy checks; these are NOT language-model accuracy."""
import itertools,json
from pathlib import Path
from roguard.evidence import plan_response
from roguard.flow import FlowLedger


def run():
    states=('absent','active','paused','revoked','expires');failures=[];count=0
    covers=[list(v for i,v in enumerate(('task','support','review')) if mask&(1<<i)) for mask in range(8)]
    report={'evidence':[{'proposed_action':'SUPPORT_AND_REVIEW'}],'review_ids':['t0_s0']}
    for state_a,state_b,cover_a,cover_b,clock in itertools.product(states,states,covers,covers,(0,2,3)):
        with FlowLedger() as ledger:
            for root,state in [('a',state_a),('b',state_b)]:
                ledger.add_source(root,'Fictional response material.')
                if state!='absent':ledger.permission(root,'tutor','minor','use','grant',expires_at=2 if state=='expires' else None)
                if state=='paused':ledger.permission(root,'tutor','minor','use','pause')
                if state=='revoked':ledger.permission(root,'tutor','minor','use','revoke')
            ledger.advance(clock)
            candidates=[{'id':'a','artifacts':['a'],'covers':cover_a,'disruption':1},{'id':'b','artifacts':['b'],'covers':cover_b,'disruption':2}]
            actual=plan_response(report,candidates,ledger,purpose='tutor',recipient='minor')
            allowed=lambda state:state=='active' or state=='expires' and clock<2
            expected='a' if allowed(state_a) and len(cover_a)==3 else 'b' if allowed(state_b) and len(cover_b)==3 else None
            count+=1
            if actual['selected']!=expected or actual['execution_authorized'] or actual['external_action_taken']:failures.append({'case':count,'expected':expected,'actual':actual})
    return {'kind':'exhaustive_declared_policy_simulation','scenarios':count,'failures':failures,'language_model_accuracy_measured':False}

if __name__=='__main__':
    result=run();path=Path('research/v8-evidence/action-simulation.json')
    if path.exists():raise ValueError('Refusing overwrite')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(bool(result['failures']))
