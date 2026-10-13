"""Repair proposals using host-declared alternatives without weakening policy."""
from copy import deepcopy
from .disclosure import minimize_disclosure
from .flow import identifiers


def repair_proposal(ledger, proposed, candidates, requirements, *, purpose, recipient, operation):
    """No generation, permission changes or execution. Coverage is host attested."""
    proposed=identifiers(proposed,'proposed artifacts')
    with ledger._lock:
        original=ledger.check(proposed,purpose,recipient,operation)
        plan=minimize_disclosure(ledger,deepcopy(candidates),requirements,purpose=purpose,recipient=recipient,operation=operation)
        eligible=plan['status']=='PROPOSAL_READY'
        return {'schema_version':'6.0','status':'REPAIR_PROPOSED' if eligible else 'NO_PERMITTED_REPAIR',
                'original_decision':original['decision'],'original_reason_codes':original['reason_codes'],
                'selected':plan['selected'],'removed_from_proposal':sorted(set(proposed)-set(plan['selected'])),
                'added_to_proposal':sorted(set(plan['selected'])-set(proposed)),
                'requirements_preserved':eligible,'semantic_sufficiency_verified':False,
                'scope':dict(purpose=purpose,recipient=recipient,operation=operation),
                'plan':plan,'execution_authorized':False,'text_in_report':False}
