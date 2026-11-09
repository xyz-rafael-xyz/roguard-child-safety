"""Canonical CPU feature arithmetic for the proposed context residual.

This defines a new inference implementation. It does not claim bit parity
with the Torch training graph. Saved-token and native callers must share it.
Only the five reader features are unavailable when a required pool is absent;
the downstream q-based correction must still be allowed to run.
"""
import math
import numpy as np

FORMAT = 'canonical-role-reader-1'
FEATURES = ('router_assistant', 'minor_choice_B', 'nonminor_choice_B',
            'minor_concern_B', 'nonminor_concern_B', 'available')
MODEL_SHAPES = {
    **{f'queries.{i}': (1024,) for i in range(3)},
    'router.weight': (2, 3584), 'router.bias': (2,),
    'choice.weight': (2, 1024), 'choice.bias': (2,),
    'concern.weight': (2, 1024), 'concern.bias': (2,),
}
STATS_SHAPES = {
    'mq': (2560,), 'mf': (1024,), 'mp': (1024,),
    'sq': (), 'sf': (), 'sp': (),
    'key_focal_mean': (1024,), 'key_focal_scale': (),
    'key_prior_mean': (1024,), 'key_prior_scale': (),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def frozen_arrays(values, shapes, dtype):
    require(set(values) == set(shapes), 'Exact parameter names required')
    result = {}
    for name, shape in shapes.items():
        a = np.asarray(values[name])
        require(a.dtype == np.dtype(dtype) and a.shape == shape,
                'Parameter dtype/shape: ' + name)
        require(np.isfinite(a).all(), 'Finite parameter: ' + name)
        a = np.array(a, copy=True, order='C')
        a.flags.writeable = False
        result[name] = a
    return result


def rms_normalize(vector):
    """F64 accumulation, norm floor1e-6, one rounded F32 vector."""
    x = np.asarray(vector, dtype=np.float64)
    norm = max(float(np.sqrt(np.sum(x*x, dtype=np.float64))), 1e-6)
    return (x/norm*math.sqrt(x.shape[0])).astype(np.float32)


class Reader:
    def __init__(self, model, stats):
        self.model = frozen_arrays(model, MODEL_SHAPES, np.float32)
        self.stats = frozen_arrays(stats, STATS_SHAPES, np.float64)
        for name in ('sq', 'sf', 'sp', 'key_focal_scale', 'key_prior_scale'):
            require(float(self.stats[name]) > 0, 'Positive scale: ' + name)

    def conditioned(self, vector, mean, scale):
        return ((rms_normalize(vector).astype(np.float64)-self.stats[mean])
                / self.stats[scale]).astype(np.float32)

    def pool(self, values, query_index, key_group):
        """F64 scores/softmax/value sum; weighted value rounded once to F32.

        Input order is the original global token order. No padding enters
        this function. Separate prior slots share queries and conditioners.
        """
        x = values.astype(np.float64)
        key = ((x-self.stats['key_'+key_group+'_mean'])
               / self.stats['key_'+key_group+'_scale'])
        query = self.model[f'queries.{query_index}'].astype(np.float64)
        scores = np.sum(key*query[None, :], axis=1, dtype=np.float64)/32.0
        weights = np.exp(scores-np.max(scores))
        weights /= np.sum(weights, dtype=np.float64)
        pooled = np.sum(weights[:, None]*x, axis=0, dtype=np.float64)
        require(np.isfinite(pooled).all(), 'Finite weighted pool')
        return pooled.astype(np.float32), weights

    def decode(self, name, features):
        # Explicit per-output F64 reduction, avoiding batch-dependent GEMMs.
        weight = self.model[name+'.weight'].astype(np.float64)
        raw = (np.sum(weight*features.astype(np.float64)[None, :],
                      axis=1, dtype=np.float64)
               + self.model[name+'.bias'].astype(np.float64)).astype(np.float32)
        require(np.isfinite(raw).all(), 'Finite decoder output')
        return raw

    def project(self, q, token_values, slots, *, detail=False):
        """slots: ordered focal, prior-minor, prior-nonminor token indices.

        The supplied tokens must be precisely the eligible content tokens;
        three disjoint slots partition them, including empty slots.
        Returns five F32 log-odds followed by a binary availability flag.
        """
        q = np.asarray(q)
        values = np.asarray(token_values)
        require(q.dtype == np.float32 and q.shape == (2560,)
                and np.isfinite(q).all(), 'Finite original F32 q')
        require(values.dtype == np.float32 and values.ndim == 2
                and len(values) <= 1024
                and values.shape[1] == 1024 and np.isfinite(values).all(),
                'Finite packed F32 token values')
        require(len(slots) == 3, 'Three actual role pools required')
        checked = []
        for slot in slots:
            indices = np.asarray(slot)
            require(indices.ndim == 1 and (indices.size == 0
                    or indices.dtype.kind in 'iu'), 'Integer token indices')
            indices = indices.astype(np.int64)
            require(np.array_equal(indices, np.unique(indices))
                    and (indices >= 0).all() and (indices < len(values)).all(),
                    'Ordered unique in-range role indices')
            checked.append(indices)
        joined = np.concatenate(checked)
        require(np.array_equal(np.sort(joined), np.arange(len(values))),
                'Disjoint exact eligible-token partition')
        if any(len(slot) == 0 for slot in checked):
            result = np.zeros(6, dtype=np.float32)
            return (result, None) if detail else result
        focal, wf = self.pool(values[checked[0]], 0, 'focal')
        router_x = np.concatenate((self.conditioned(q, 'mq', 'sq'),
                                   self.conditioned(focal, 'mf', 'sf')))
        raw = [self.decode('router', router_x)]
        attention = {'router': wf, 'choice': [], 'concern': []}
        for name, query_index in (('choice', 1), ('concern', 2)):
            for slot in checked[1:]:
                pooled, weights = self.pool(values[slot], query_index, 'prior')
                raw.append(self.decode(name, self.conditioned(pooled, 'mp', 'sp')))
                attention[name].append(weights)
        logits = np.stack(raw)
        features = np.empty(6, dtype=np.float32)
        features[:5] = (logits[:, 1].astype(np.float64)
                        - logits[:, 0].astype(np.float64)).astype(np.float32)
        features[5] = 1.0
        require(np.isfinite(features).all(), 'Finite reader features')
        return (features, {'raw2': logits, 'attention': attention}) if detail else features
