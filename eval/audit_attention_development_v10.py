"""Independent recount of saved binary-attention DEV outputs; no encoder inference."""
import math


def threshold_audit(rows, scores):
    """Enumerate attainable cuts using sorted group counts, without trainer selectors."""
    import numpy as np
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(rows),) or not len(rows) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid score population')
    gold = np.array([r['gold'] for r in rows])
    if any(type(r['gold']) is not int or r['gold'] not in (0, 1)
           or not isinstance(r['group'], str) or not r['group'] for r in rows):
        raise ValueError('Invalid labels or groups')
    unique = sorted(set(values.tolist()))
    cuts = [unique[0]/2]
    for lower, upper in zip(unique, unique[1:]):
        cut = lower+(upper-lower)/2
        cuts.append(cut if lower < cut < upper else upper)
    if unique[-1] < 1:
        cut = unique[-1]+(1-unique[-1])/2
        cuts.append(cut if cut > unique[-1] else 1.)
    risks, errors = [], np.zeros(len(cuts), dtype=np.int64)
    for group in sorted({row['group'] for row in rows}):
        mask = np.array([row['group'] == group for row in rows])
        positive = np.sort(values[mask & (gold == 1)])
        negative = np.sort(values[mask & (gold == 0)])
        if not len(positive) or not len(negative):
            raise ValueError('Each group needs both classes')
        missed = np.searchsorted(positive, cuts, side='left')
        false = len(negative)-np.searchsorted(negative, cuts, side='left')
        risks.extend((100*missed/len(positive), 20*false/len(negative)))
        errors += missed+false
    # Python summation follows the registered group order, including floating ties.
    ranks = [(max(risk), sum(risk), int(count)) for risk, count in zip(zip(*[r.tolist() for r in risks]), errors)]
    index = min(range(len(cuts)), key=lambda i: ranks[i])
    threshold, margin = cuts[index], 0.
    if ranks[index] == (0., 0., 0):
        lower = float(values[gold == 0].max()); upper = float(values[gold == 1].min())
        threshold = lower+(upper-lower)/2
        if not lower < threshold <= upper:
            threshold = upper
        margin = min(threshold-lower, upper-threshold)
    return {'threshold': threshold, 'score_margin': margin, 'errors': int(errors[index]),
            'minimax_rank': list(ranks[index])}


def enumeration_scores(factors, couplings=None):
    """Compute policy probability by summing all ontology worlds, not the closed form."""
    import itertools
    import numpy as np
    from roguard.evidence import ONTOLOGY, disposition
    fields = tuple(ONTOLOGY)
    worlds = list(itertools.product(*(range(len(ONTOLOGY[f])) for f in fields)))
    action = np.array([disposition({f: {'label': ONTOLOGY[f][v]} for f, v in zip(fields, world)})[0] != 'CONTINUE'
                       for world in worlds], dtype=np.float64)
    n = len(factors[0]); result = np.empty(n, dtype=np.float64)
    for start in range(0, n, 128):
        probability = np.ones((min(128, n-start), len(worlds)), dtype=np.float64)
        for j, values in enumerate(factors):
            probability *= values[start:start+128, [w[j] for w in worlds]]
        if couplings is not None:
            for name, left, right in (('scope', 'subject', 'source'), ('assertion_relation', 'assertion', 'relation')):
                i, j = fields.index(left), fields.index(right)
                probability *= np.asarray(couplings[name])[[w[i] for w in worlds], [w[j] for w in worlds]][None, :]
        result[start:start+128] = (probability*action).sum(axis=1)/probability.sum(axis=1)
    return result


def recount(rows, logits, couplings, baseline):
    """Reproduce all three mode choices, field metrics and paired decisions."""
    import numpy as np
    from roguard.evidence import ONTOLOGY
    from training.native_fields_v9 import policy_mass
    sizes = tuple(map(len, ONTOLOGY.values()))
    z = np.asarray(logits, dtype=np.float64)
    gold = np.array([r['gold'] for r in rows], dtype=int)
    labels = np.array([r['fields'] for r in rows], dtype=int)
    baseline = np.asarray(baseline)
    if (z.shape != (len(rows), sum(sizes)+2) or not np.isfinite(z).all()
            or baseline.shape != gold.shape or not np.isin(baseline, [0, 1]).all()
            or labels.shape != (len(rows), len(sizes))
            or any(((labels[:, j] < 0) | (labels[:, j] >= size)).any() for j, size in enumerate(sizes))):
        raise ValueError('Malformed saved development arrays')
    def softmax(v):
        p = np.exp(v-v.max(axis=1, keepdims=True))
        return p/p.sum(axis=1, keepdims=True)
    factors = [softmax(v) for v in np.split(z[:, :sum(sizes)], np.cumsum(sizes)[:-1], axis=1)]
    modes = {'binary': softmax(z[:, -2:])[:, 1], 'policy': policy_mass(factors),
             'coupled_policy': policy_mass(factors, couplings)}
    maximum_policy_error = 0.
    for mode, table in (('policy', None), ('coupled_policy', couplings)):
        error = float(np.max(np.abs(modes[mode]-enumeration_scores(factors, table))))
        maximum_policy_error = max(error, maximum_policy_error)
        if not math.isfinite(error) or error > 1e-12:
            raise ValueError('Closed-form scores differ from independent world enumeration')
    choices = [(mode, threshold_audit(rows, score)) for mode, score in modes.items()]
    mode, choice = min(choices, key=lambda item: (item[1]['errors'], -item[1]['score_margin']))
    predictions = (modes[mode] >= choice['threshold']).astype(int)
    field_predictions = np.column_stack([v.argmax(axis=1) for v in factors])
    mismatch = field_predictions != labels
    ce = float(np.mean([-np.log(np.maximum(values[np.arange(len(rows)), labels[:, j]], 1e-30)).mean()
                        for j, values in enumerate(factors)]))
    joint = int(mismatch.any(axis=1).sum())
    groups = {}
    for group in sorted({row['group'] for row in rows}):
        mask = np.array([row['group'] == group for row in rows])
        groups[group] = {'rows': int(mask.sum()), 'errors': int(((predictions != gold) & mask).sum()),
                         'introduced': int(((predictions != gold) & (baseline == gold) & mask).sum())}
    summary = {'binary_errors': int((predictions != gold).sum()), 'joint_argmax_errors': joint,
               'field_errors': mismatch.sum(axis=0).tolist(), 'mean_field_cross_entropy': ce,
               'mode': mode, 'threshold': choice['threshold'], 'score_margin': choice['score_margin'],
               'selection_rank': [choice['errors'], joint, ce, -choice['score_margin']],
               'fixed_v9': int(((predictions == gold) & (baseline != gold)).sum()),
               'introduced_v9': int(((predictions != gold) & (baseline == gold)).sum()), 'groups': groups}
    return summary, predictions, modes[mode], maximum_policy_error


def verify_checkpoint(rows, logits, couplings, baseline, record, *, epoch, train_rows, batch_size):
    """Fail closed on counters and any saved statistic, including nonselected epochs."""
    summary, predictions, scores, error = recount(rows, logits, couplings, baseline)
    if (record['epoch'] != epoch or record['completed_training_updates'] != epoch*((train_rows+batch_size-1)//batch_size)
            or record['completed_training_presentations'] != epoch*train_rows):
        raise ValueError('Checkpoint counters differ from the registered completed epoch')
    for key, value in summary.items():
        if record.get(key) != value:
            raise ValueError('Saved DEV statistic differs: '+key)
    return predictions, scores, error
