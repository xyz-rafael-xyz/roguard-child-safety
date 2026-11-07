"""Input, objective and selection helpers for streaming binary attention research."""
from collections import Counter

from roguard.assurance import digest
from roguard.gemma4_contract import PREFIX_TOKENS
from training.native_fields_v9 import SIZES, policy_mass
from training.separation_frontier_v9 import frontier


def validate_rows(recovered, metadata):
    """Bind recovered source inputs to exact feature positions, never just IDs."""
    seen = set()
    result = {language: [None] * len(rows) for language, rows in metadata.items()}
    extra = {'feature_index', 'payload', 'tokens', 'token_count', 'source_name',
             'source_row_sha256', 'source_focus'}
    for row in recovered:
        lang, index = row['language'], row['feature_index']
        if lang not in result or type(index) is not int or not 0 <= index < len(result[lang]):
            raise ValueError('Invalid recovered feature position')
        if result[lang][index] is not None or {k: v for k, v in row.items() if k not in extra} != metadata[lang][index]:
            raise ValueError('Recovered metadata differs from its feature position')
        if (row['partition'] not in ('train', 'dev') or type(row['gold']) is not int
                or row['gold'] not in (0, 1) or len(row['fields']) != len(SIZES)
                or any(type(v) is not int or not 0 <= v < n for v, n in zip(row['fields'], SIZES))):
            raise ValueError('Invalid supervision or partition')
        tokens = row['tokens']
        if (not isinstance(tokens, list) or any(type(v) is not int or v < 0 for v in tokens)
                or not len(PREFIX_TOKENS) < len(tokens) <= 1024
                or tokens[:len(PREFIX_TOKENS)] != PREFIX_TOKENS
                or row['token_count'] != len(tokens) or digest(tokens) != row['token_sha256']):
            raise ValueError('Recovered tokens differ from the runtime input')
        if row['payload']['language'] != lang or not row['payload']['turns']:
            raise ValueError('Invalid recovered payload')
        identity = (lang, row['token_sha256'])
        if identity in seen:
            raise ValueError('Repeated token input across partitions or within a population')
        seen.add(identity)
        result[lang][index] = row
    for lang, rows in result.items():
        if any(row is None for row in rows) or len({row['case_id'] for row in rows}) != len(rows):
            raise ValueError('Incomplete or repeated recovered input')
        if set(row['partition'] for row in rows) != {'train', 'dev'}:
            raise ValueError('Expected both TRAIN and DEV')
    return result


def field_weights(labels):
    """Compute class weights from TRAIN only, matching the preceding head fit."""
    import numpy as np
    result = []
    for field, size in enumerate(SIZES):
        counts = np.bincount(labels[:, field], minlength=size)
        weights = np.minimum(4., np.sqrt(len(labels) / np.maximum(1, counts) / size))
        weights /= np.mean(weights[labels[:, field]])
        result.append(weights.astype(np.float32))
    return result


def make_loss(mx, nn, couplings, weights, *, teacher_kl, policy_ce):
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass, policy_binary_cross_entropy
    score = make_coupled_policy_mass(mx, couplings)

    def loss(logits, gold, fields, teacher):
        offset = 0
        terms, divergences = [], []
        for j, size in enumerate(SIZES):
            values, old = logits[:, offset:offset+size], teacher[:, offset:offset+size]
            terms.append(nn.losses.cross_entropy(values, fields[:, j]) * weights[j][fields[:, j]])
            old_logp = old - mx.logsumexp(old, axis=-1, keepdims=True)
            logp = values - mx.logsumexp(values, axis=-1, keepdims=True)
            divergences.append((mx.exp(old_logp) * (old_logp-logp)).sum(axis=-1))
            offset += size
        values, old = logits[:, -2:], teacher[:, -2:]
        old_logp = old - mx.logsumexp(old, axis=-1, keepdims=True)
        logp = values - mx.logsumexp(values, axis=-1, keepdims=True)
        divergences.append((mx.exp(old_logp) * (old_logp-logp)).sum(axis=-1))
        per_row = (nn.losses.cross_entropy(values, gold) + mx.stack(terms).mean(axis=0)
                   + teacher_kl * mx.stack(divergences).mean(axis=0))
        if policy_ce:
            per_row += policy_ce * policy_binary_cross_entropy(mx, score(logits[:, :sum(SIZES)]), gold)
        return per_row.mean()
    return loss


def summarize(rows, logits, couplings, baseline_predictions):
    """DEV-only selection; all original-v9 comparisons use its fixed decisions."""
    import numpy as np
    z = np.asarray(logits, dtype=np.float64)
    if z.shape != (len(rows), sum(SIZES)+2) or not np.isfinite(z).all():
        raise ValueError('Invalid development logits')
    labels = np.array([row['fields'] for row in rows], dtype=np.int32)
    gold = np.array([row['gold'] for row in rows], dtype=np.int32)
    baseline = np.asarray(baseline_predictions)
    if baseline.shape != gold.shape or not np.isin(baseline, [0, 1]).all():
        raise ValueError('Invalid fixed baseline predictions')
    def softmax(v):
        p = np.exp(v-v.max(axis=1, keepdims=True))
        return p/p.sum(axis=1, keepdims=True)
    factors = [softmax(v) for v in np.split(z[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)]
    predictions = np.column_stack([v.argmax(axis=1) for v in factors])
    joint = int(np.any(predictions != labels, axis=1).sum())
    ce = float(np.mean([-np.log(np.maximum(v[np.arange(len(rows)), labels[:, j]], 1e-30)).mean()
                        for j, v in enumerate(factors)]))
    choices = []
    for mode, scores in (('binary', softmax(z[:, -2:])[:, 1]), ('policy', policy_mass(factors)),
                         ('coupled_policy', policy_mass(factors, couplings))):
        population = [{'case_id': r['case_id'], 'language': r['language'], 'group': r['group'],
                       'gold': int(r['gold']), 'score': float(score)} for r, score in zip(rows, scores)]
        boundary = frontier(population)
        binary = (scores >= boundary['threshold']).astype(int)
        errors = int((binary != gold).sum())
        choices.append((errors, -boundary['score_margin'], mode, boundary, population, binary))
    errors, negative_margin, mode, boundary, population, binary = min(choices, key=lambda c: c[:2])
    for r, p, b in zip(population, binary, baseline):
        r.update(predicted=int(p), baseline_v9=int(b))
    introduced = int(((binary != gold) & (baseline == gold)).sum())
    fixed = int(((binary == gold) & (baseline != gold)).sum())
    groups = {}
    for group in sorted(set(row['group'] for row in rows)):
        mask = np.array([row['group'] == group for row in rows])
        groups[group] = {'rows': int(mask.sum()), 'errors': int(((binary != gold) & mask).sum()),
                         'introduced': int(((binary != gold) & (baseline == gold) & mask).sum())}
    return {'binary_errors': errors, 'joint_argmax_errors': joint,
            'field_errors': (predictions != labels).sum(axis=0).tolist(),
            'mean_field_cross_entropy': ce, 'mode': mode, 'threshold': boundary['threshold'],
            'score_margin': -negative_margin, 'selection_rank': [errors, joint, ce, negative_margin],
            'fixed_v9': fixed, 'introduced_v9': introduced, 'groups': groups}, population


def check_trainable(model, language, *, boundary=24):
    from mlx.utils import tree_flatten
    from roguard.gemma4_attention import _backbone
    body = _backbone(model.base, boundary)
    names = {f'base.model.layers.{i}.self_attn.{p}.lora_{f}'
             for i in range(boundary, len(body.layers)) for p in ('q_proj', 'o_proj') for f in ('a', 'b')}
    names |= {f'heads.{language}.{layer}.{part}' for layer in ('first', 'last') for part in ('weight', 'bias')}
    if set(dict(tree_flatten(model.trainable_parameters()))) != names:
        raise ValueError('Unregistered trainable parameter inventory')
    return names


def make_optimizer(config, language):
    import mlx.optimizers as optim
    tail = optim.AdamW(learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])
    head = optim.AdamW(learning_rate=config['head_learning_rate'], weight_decay=config['weight_decay'])
    return (optim.MultiOptimizer([head, tail], [lambda name, _: name.startswith(f'heads.{language}.')]),
            [(head, config['head_learning_rate']), (tail, config['learning_rate'])])


def training_schedule(rows, epochs, batch_size):
    counts = Counter(row['partition'] for row in rows)
    return {'updates_per_epoch': (counts['train']+batch_size-1)//batch_size,
            'total_updates': epochs*((counts['train']+batch_size-1)//batch_size),
            'total_presentations': epochs*counts['train']}
