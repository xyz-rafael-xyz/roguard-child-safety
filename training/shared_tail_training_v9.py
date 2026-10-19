"""Learned semantic heads and matched-history losses over a shared-KV tail."""
from collections import defaultdict

from training.native_fields_v9 import SIZES


def plan_batches(rows, rng, size=8):
    families, ordinary, singles = defaultdict(list), defaultdict(list), []
    for i, row in enumerate(rows):
        if row.get('context_family'):
            families[row['context_family']].append(i)
        elif row.get('pair_group'):
            ordinary[(row['source'], row['pair_group'])].append(i)
        else:
            singles.append([i])
    if any(len(v) != 6 for v in families.values()):
        raise ValueError('Every matched-history family must have six cases')
    units = list(families.values())+singles
    for group in ordinary.values():
        units.extend([group] if len(group) == 2 else [[i] for i in group])
    rng.shuffle(units)
    batches, pending = [], []
    for unit in units:
        if len(unit) > size:
            raise ValueError('Supervision family exceeds batch size')
        if len(pending)+len(unit) > size:
            batches.append(pending)
            pending = []
        pending.extend(unit)
    if pending:
        batches.append(pending)
    result = []
    for indices in batches:
        pairs, invariant = [], []
        for a, i in enumerate(indices):
            for b in range(a+1, len(indices)):
                j = indices[b]
                left, right = rows[i], rows[j]
                if left.get('pair_group') and left['pair_group'] == right.get('pair_group'):
                    from training.prepare_instruction_sft_v8 import target
                    y0, y1 = target(left), target(right)
                    if y0 != y1:
                        pairs.append((a, b) if y0 == 0 else (b, a))
                if left.get('invariance_group') and left['invariance_group'] == right.get('invariance_group'):
                    if left['attributes'] != right['attributes'] or left['turns'][left['focus']] != right['turns'][right['focus']]:
                        raise ValueError('History invariance requires identical focal text and labels')
                    invariant.append((a, b))
        result.append((indices, pairs, invariant))
    return result


def make_head(tensors):
    import mlx.core as mx
    import mlx.nn as nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            hidden, dimensions = tensors['0.weight'].shape
            self.first = nn.Linear(dimensions, hidden)
            self.last = nn.Linear(hidden, sum(SIZES)+2)
            self.first.weight, self.first.bias = mx.array(tensors['0.weight']), mx.array(tensors['0.bias'])
            self.last.weight, self.last.bias = mx.array(tensors['3.weight']), mx.array(tensors['3.bias'])
        def __call__(self, features):
            normalized = features/mx.sqrt(mx.sum(mx.square(features), axis=1, keepdims=True))*features.shape[1]**.5
            return self.last(nn.gelu(self.first(normalized)))
    return Head()


def make_classifier(base, tensors, *, boundary, lora):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from training.gemma4_shared_tail_v9 import _backbone, replay
    from training.shared_tail_cache_v9 import batch_states
    _backbone(base, boundary)
    base.freeze()
    linear_to_lora_layers(base, len(base.layers)-boundary, lora)
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.heads = {language: make_head(head) for language, head in tensors.items()}
        def __call__(self, states, languages):
            if len(states) != len(languages) or not states:
                raise ValueError('Mismatched cached classifier inputs')
            grouped = defaultdict(list)
            for i, (state, language) in enumerate(zip(states, languages)):
                if language not in self.heads or state['start'] != boundary:
                    raise ValueError('Unexpected language or tail boundary')
                grouped[(language, state['length'])].append(i)
            output = [None]*len(states)
            for (language, _), indices in grouped.items():
                hidden = replay(self.base, batch_states([states[i] for i in indices]))[:, 0, :]
                values = self.heads[language](hidden)
                for position, index in enumerate(indices):
                    output[index] = values[position]
            return mx.stack(output)
    return Classifier()


def supervised_loss(logits, review, fields, weights, pairs, invariant):
    import mlx.core as mx
    import mlx.nn as nn
    pieces, offset = [], 0
    for size in SIZES:
        pieces.append(logits[:, offset:offset+size])
        offset += size
    binary = logits[:, -2:]
    terms = [nn.losses.cross_entropy(values, fields[:, i]) for i, values in enumerate(pieces)]
    value = ((nn.losses.cross_entropy(binary, review)+mx.stack(terms).mean(axis=0))*weights).mean()
    if pairs:
        margin = binary[:, 1]-binary[:, 0]
        negative, positive = [a for a, b in pairs], [b for a, b in pairs]
        value = value+.2*mx.logaddexp(0., 1.-(margin[positive]-margin[negative])).mean()
    if invariant:
        left, right = [a for a, b in invariant], [b for a, b in invariant]
        divergences = []
        for values in pieces+[binary]:
            logp = values-mx.logsumexp(values, axis=-1, keepdims=True)
            p = mx.exp(logp)
            divergences.append(((p[left]-p[right])*(logp[left]-logp[right])).sum(axis=-1).mean()/2)
        value = value+.1*mx.stack(divergences).mean()
    return value
