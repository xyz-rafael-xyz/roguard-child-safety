"""Adaptive binary-policy integration over a conditional interpretation tree.

Bounds describe the supplied constrained model distribution, not calibrated
real-world confidence. Unqueried branches retain all their mass. A branch can
be collapsed only when every ontology completion has the same review action.
"""
from functools import lru_cache
import heapq
import itertools
import math

from roguard.evidence import ONTOLOGY, disposition
from training.semantic_interpreter_v9 import FIELDS


@lru_cache(maxsize=None)
def actions(prefix):
    result = set()
    for tail in itertools.product(*(ONTOLOGY[f] for f in FIELDS[len(prefix):])):
        world = dict(zip(FIELDS, prefix + tail))
        result.add(int(disposition({k: {'label': v} for k, v in world.items()})[0] != 'CONTINUE'))
    return frozenset(result)


def decode(conditional, *, threshold=0.5, max_queries=64, tolerance=0.0):
    """Query largest ambiguous mass first; never renormalize pruned mass.

    conditional(prefix, field) returns probabilities in ONTOLOGY[field] order,
    or None for an invalid model output. Such a branch remains unresolved.
    The decision is 1 iff probability >= threshold. Exact equality is positive.
    """
    if type(max_queries) is not int or not 0 <= max_queries <= 1024:
        raise ValueError('Invalid query budget')
    if not 0 < threshold < 1 or not math.isfinite(threshold):
        raise ValueError('Invalid threshold')
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError('Invalid tolerance')
    queue = [(-1.0, ())]
    positive = negative = invalid = 0.0
    queries = collapsed = 0
    leaves = []
    while queue:
        # Include a numerical cushion when certifying a threshold crossing.
        lower = max(0.0, positive - 1e-12)
        upper = min(1.0, 1.0 - negative + 1e-12)
        if lower >= threshold or upper < threshold or upper-lower <= tolerance:
            break
        neg_mass, prefix = heapq.heappop(queue)
        mass = -neg_mass
        possible = actions(prefix)
        if len(possible) == 1:
            value = next(iter(possible))
            if value: positive += mass
            else: negative += mass
            leaves.append(dict(prefix=list(prefix), mass=mass, review=value))
            collapsed += 1
            continue
        if queries >= max_queries:
            heapq.heappush(queue, (neg_mass, prefix))
            break
        field = FIELDS[len(prefix)]
        probabilities = conditional(prefix, field)
        queries += 1
        if probabilities is None:
            invalid += mass
            continue
        if len(probabilities) != len(ONTOLOGY[field]) or any(
                type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
                for p in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError('Conditional probabilities must sum to one')
        # Normalize only numerical summation error in a complete conditional.
        total = sum(probabilities)
        for label, probability in zip(ONTOLOGY[field], probabilities):
            if probability:
                heapq.heappush(queue, (-mass*probability/total, prefix+(label,)))
    residual = sum(-mass for mass, prefix in queue) + invalid
    if not math.isclose(positive+negative+residual, 1.0, abs_tol=1e-9):
        raise AssertionError('Probability mass was lost')
    if residual == 0:
        lower = upper = positive
    else:
        lower = max(0.0, positive-1e-12)
        upper = min(1.0, positive+residual+1e-12)
    decision = 1 if lower >= threshold else 0 if upper < threshold else None
    return dict(predicted=decision, lower_review_mass=lower, upper_review_mass=upper,
                unresolved_mass=residual, invalid_mass=invalid, queries=queries,
                collapsed_branches=collapsed, resolved=leaves,
                probabilities_are_calibrated=False, external_action_taken=False)
