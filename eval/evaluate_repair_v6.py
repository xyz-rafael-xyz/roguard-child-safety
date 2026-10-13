"""Controlled repair evaluation independent of generative-model claims."""
import json,argparse
from pathlib import Path
from roguard.flow import FlowLedger
from roguard.repair import repair_proposal

def evaluate():
    counts=dict(cases=0,original_blocked=0,repaired_permitted=0,correctly_unrepairable=0,unauthorized_repairs=0,revocations_stopped=0)
    for lang in ('ro','uk'):
        for seed in range(100):
            for possible in (False,True):
                with FlowLedger() as ledger:
                    for name in ('learning','personal'):ledger.add_source(name,'Exemplu fictiv' if lang=='ro' else 'Вигаданий приклад')
                    if possible:ledger.permission('learning','lesson','teacher','export','grant')
                    args=dict(proposed=['learning','personal'],candidates=[dict(artifact_id='learning',covers=['goal'],cost=1+seed%4),dict(artifact_id='personal',covers=['goal'],cost=10)],requirements=['goal'],purpose='lesson',recipient='teacher',operation='export')
                    result=repair_proposal(ledger,**args);counts['cases']+=1;counts['original_blocked']+=result['original_decision']=='BLOCK'
                    if result['selected']:
                        check=ledger.check(result['selected'],'lesson','teacher','export');counts['repaired_permitted']+=check['decision']=='ALLOW';counts['unauthorized_repairs']+=check['decision']!='ALLOW'
                        ledger.permission('learning','lesson','teacher','export','revoke');counts['revocations_stopped']+=ledger.check(result['selected'],'lesson','teacher','export')['decision']=='BLOCK'
                    else:counts['correctly_unrepairable']+=not possible
    return dict(**counts,synthetic_reference_only=True,variants_share_templates=True,semantic_sufficiency_verified=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(evaluate(),f,indent=2)
