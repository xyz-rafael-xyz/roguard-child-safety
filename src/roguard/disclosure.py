"""Exact disclosure selection under declared coverage and privacy costs."""
from __future__ import annotations
from .flow import identifiers, integer, scope


def _minimize_disclosure(ledger, candidates, requirements, *, purpose, recipient, operation):
    """Return an optimal proposal within <=18 declared candidates, never content.

    Objective: total declared cost, then field count, then lexical identifier
    order. Coverage declarations are not semantic verification of a summary.
    Execution must recheck the live ledger; this proposal is not a capability.
    """
    scope(purpose, recipient, operation)
    requirements = identifiers(requirements, 'requirements', maximum=16)
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 18:
        raise ValueError('Provide 1 to 18 candidates')
    names = []; eligible = []; excluded = {}; declared = {}
    bits = {name: 1 << i for i, name in enumerate(requirements)}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {'artifact_id', 'covers', 'cost'}:
            raise ValueError('Candidates require artifact_id, covers and cost')
        name = identifiers([candidate['artifact_id']])[0]
        covers = identifiers(candidate['covers'], 'coverage', maximum=16, allow_empty=True)
        cost = integer(candidate['cost'], 'disclosure cost', 10**6)
        if name in names:
            raise ValueError('Duplicate candidate')
        names.append(name); declared[name] = covers
        check = ledger.check([name], purpose, recipient, operation)
        mask = sum(bits.get(goal, 0) for goal in covers)
        if check['decision'] != 'ALLOW':
            excluded[name] = check['reason_codes']
        elif mask:
            eligible.append((name, mask, cost))
        else:
            excluded[name] = ['NO_REQUIRED_COVERAGE']
    # A mask stores the lexicographically optimal objective tuple.
    best = {0: (0, 0, ())}
    for name, covered, cost in sorted(eligible):
        for mask, old in list(best.items()):
            combined = mask | covered
            option = (old[0] + cost, old[1] + 1, tuple(sorted(old[2] + (name,))))
            if combined not in best or option < best[combined]:
                best[combined] = option
    target = (1 << len(requirements)) - 1
    optimal = best.get(target)
    selected = list(optimal[2]) if optimal else []
    available = 0
    for _, mask, _ in eligible:
        available |= mask
    return {'status': 'PROPOSAL_READY' if optimal else 'INSUFFICIENT_PERMITTED_COVERAGE',
            'selected': selected, 'declared_cost': optimal[0] if optimal else None,
            'retained': [{'artifact_id': name, 'requirements': sorted(set(declared[name]) & set(requirements))}
                         for name in selected],
            'removed': [{'artifact_id': name, 'reason_codes': excluded.get(name, ['NOT_NEEDED_FOR_OPTIMAL_PROPOSAL'])}
                        for name in sorted(set(names) - set(selected))],
            'missing_requirements': [name for name, bit in bits.items() if not bit & available],
            'optimal_within_declared_candidates': bool(optimal), 'objective': 'cost_then_count_then_ids',
            'policy_revision': ledger.revision, 'clock': ledger.clock,
            'semantic_sufficiency_verified': False, 'execution_authorized': False,
            'text_in_report': False}


def minimize_disclosure(ledger, candidates, requirements, *, purpose, recipient, operation):
    """Optimize against one locked policy revision; execution still rechecks."""
    from .flow import FlowLedger
    if not isinstance(ledger, FlowLedger):
        raise ValueError('Expected FlowLedger')
    with ledger._lock:
        return _minimize_disclosure(ledger, candidates, requirements, purpose=purpose,
                                   recipient=recipient, operation=operation)
