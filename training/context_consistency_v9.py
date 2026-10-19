"""Matched-history minibatches and consistency loss for the two NLI reasons."""
from collections import defaultdict

from training.policy_entailment_v9 import targets


def plan_batches(rows, rng):
    groups = defaultdict(list)
    singles = []
    for i, row in enumerate(rows):
        key = row.get('context_family') or row.get('pair_group')
        if key:
            groups[(row['source'], key)].append(i)
        else:
            singles.append([i])
    units = list(groups.values())+singles
    if any(len(unit) > 6 for unit in units):
        raise ValueError('A supervision family exceeds the six-row batch')
    rng.shuffle(units)
    batches, pending = [], []
    for unit in units:
        if len(pending)+len(unit) > 6:
            batches.append(pending)
            pending = []
        pending.extend(unit)
    if pending:
        batches.append(pending)
    result = []
    for original in batches:
        indices = [2*i+t for i in original for t in (0, 1)]
        positions = {value: i for i, value in enumerate(indices)}
        pairs, invariant = [], []
        for a, i in enumerate(original):
            for j in original[a+1:]:
                left, right = rows[i], rows[j]
                same_pair = left.get('pair_group') and left.get('pair_group') == right.get('pair_group')
                same_history_family = left.get('invariance_group') and left.get('invariance_group') == right.get('invariance_group')
                for task in (0, 1):
                    t0, t1 = targets(left)[task], targets(right)[task]
                    if same_pair and t0 != t1:
                        negative, positive = (i, j) if t0 == 0 else (j, i)
                        pairs.append((positions[2*negative+task], positions[2*positive+task]))
                    if same_history_family:
                        if t0 != t1 or left['turns'][left['focus']] != right['turns'][right['focus']]:
                            raise ValueError('Invariance requires identical current messages and targets')
                        invariant.append((positions[2*i+task], positions[2*j+task]))
        result.append((indices, pairs, invariant))
    return result


def loss(logits, labels, weights, pairs, invariant):
    import torch
    import torch.nn.functional as F
    value = (F.cross_entropy(logits, labels, reduction='none')*weights).mean()
    if pairs:
        margin = logits[:, 0]-torch.logsumexp(logits[:, 1:], dim=-1)
        value = value+.2*F.softplus(1.-(margin[[p for n, p in pairs]]-margin[[n for n, p in pairs]])).mean()
    if invariant:
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        left, right = [a for a, b in invariant], [b for a, b in invariant]
        divergence = ((p[left]-p[right])*(logp[left]-logp[right])).sum(dim=-1).mean()/2
        value = value+.1*divergence
    return value
