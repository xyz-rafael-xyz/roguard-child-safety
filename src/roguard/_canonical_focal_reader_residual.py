"""Canonical inference for an unmasked q+reader+focal correction to original v9.

This module is the prospective inference definition, not a claim of bit
parity with the Torch optimization graph. The v9 parent logits are inputs.
Missing role pools neutralize only the five reader features; q corrections
remain possible. No labels, source-bank identifiers, or text parsers enter.
"""
import math
import numpy as np

FORMAT = 'canonical-focal-reader-residual19-1'
WIDTH = 3590
HIDDEN = 128
MODEL_SHAPES = {'first.weight': (128, 3590), 'first.bias': (128,),
                'last.weight': (19, 128), 'last.bias': (19,)}
STATS_SHAPES = {'q_mean': (2560,), 'q_scale': (),
                'reader_mean': (5,), 'reader_scale': (),
                'focal_mean': (1024,), 'focal_scale': ()}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def copy_arrays(values, shapes, dtype):
    require(set(values) == set(shapes), 'Exact parameter names')
    result = {}
    for key, shape in shapes.items():
        a = np.asarray(values[key])
        require(a.shape == shape and a.dtype == np.dtype(dtype) and np.isfinite(a).all(),
                'Finite exact dtype/shape: ' + key)
        a = np.array(a, order='C', copy=True)
        a.flags.writeable = False
        result[key] = a
    return result


def normalized_q(q):
    require(q.dtype == np.float32 and q.shape == (2560,) and np.isfinite(q).all(),
            'Finite original F32 q')
    x = q.astype(np.float64)
    norm = max(float(np.sqrt(np.sum(x*x, dtype=np.float64))), 1e-6)
    return (x/norm*math.sqrt(2560)).astype(np.float32)


def normalized_focal(focal):
    require(focal.dtype == np.float32 and focal.shape == (1024,)
            and np.isfinite(focal).all(), 'Finite F32 focal mean')
    x = focal.astype(np.float64)
    norm = max(float(np.sqrt(np.sum(x*x, dtype=np.float64))), 1e-6)
    return (x/norm*math.sqrt(1024)).astype(np.float32)


class Conditioner:
    def __init__(self, stats):
        self.stats = copy_arrays(stats, STATS_SHAPES, np.float64)
        require(float(self.stats['q_scale']) > 0 and float(self.stats['reader_scale']) > 0
                and float(self.stats['focal_scale']) > 0,
                'Positive TRAIN-only conditioning scales')

    def prepare(self, q, reader, focal, focal_present):
        require(reader.dtype == np.float32 and reader.shape == (6,)
                and np.isfinite(reader).all() and reader[5] in (0., 1.),
                'Five finite F32 reader log-odds plus availability bit')
        require(reader[5] == 1 or (reader[:5] == 0).all(),
                'Unavailable reader features must be explicit neutral zeros')
        x = np.zeros(WIDTH, np.float32)
        x[:2560] = ((normalized_q(q).astype(np.float64)-self.stats['q_mean'])
                    / self.stats['q_scale']).astype(np.float32)
        if reader[5] == 1:
            x[2560:2565] = ((reader[:5].astype(np.float64)-self.stats['reader_mean'])
                            / self.stats['reader_scale']).astype(np.float32)
        x[2565] = reader[5]
        require(type(focal_present) is bool, 'Explicit focal token availability')
        f = normalized_focal(focal)
        require(focal_present or not np.count_nonzero(focal),
                'Absent focal pool must be exact neutral raw zeros')
        if focal_present:
            x[2566:] = ((f.astype(np.float64)-self.stats['focal_mean'])
                        / self.stats['focal_scale']).astype(np.float32)
        require(np.isfinite(x).all(), 'Finite conditioned residual inputs')
        return x


class Residual:
    def __init__(self, weights, stats):
        self.weights = copy_arrays(weights, MODEL_SHAPES, np.float32)
        self.conditioner = Conditioner(stats)
        self.double = {k: a.astype(np.float64) for k, a in self.weights.items()}
        for a in self.double.values():
            a.flags.writeable = False
        self.zero_output = not (np.count_nonzero(self.weights['last.weight'])
                                or np.count_nonzero(self.weights['last.bias']))

    def predict(self, q, reader, parent_raw19, *, focal, focal_present, enabled=True):
        require(parent_raw19.dtype == np.float32 and parent_raw19.shape == (19,)
                and np.isfinite(parent_raw19).all(), 'Finite original v9 raw19')
        if not enabled:
            return parent_raw19.copy()
        x = self.conditioner.prepare(q, reader, focal, focal_present)
        # Preserve even signed-zero parent bytes at the registered zero endpoint.
        if self.zero_output:
            return parent_raw19.copy()
        h = np.sum(self.double['first.weight']*x.astype(np.float64)[None, :],
                   axis=1, dtype=np.float64)+self.double['first.bias']
        require(np.isfinite(h).all(), 'Finite first layer')
        gelu = np.fromiter((.5*float(t)*(1+math.erf(float(t)/math.sqrt(2))) for t in h),
                           dtype=np.float64, count=HIDDEN)
        delta = (np.sum(self.double['last.weight']*gelu[None, :], axis=1,
                        dtype=np.float64)+self.double['last.bias']).astype(np.float32)
        result = np.add(parent_raw19, delta, dtype=np.float32)
        require(np.isfinite(result).all(), 'Finite corrected raw19')
        return result
