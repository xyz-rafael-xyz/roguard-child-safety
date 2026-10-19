"""Portable radial-basis classifier over frozen contextual representations."""
import math
from training.gemma4_readout_v9 import normalize


def margins(features, head, support):
    import numpy as np
    values = normalize(features)
    vectors = np.asarray(support, dtype=np.float64)
    coefficients = np.asarray(head['dual_coefficients'], dtype=np.float64)
    gamma, intercept = head['gamma'], head['intercept']
    if (vectors.ndim != 2 or vectors.shape[1] != values.shape[1]
            or not vectors.shape[0] or coefficients.shape != (vectors.shape[0],)
            or not np.isfinite(vectors).all() or not np.isfinite(coefficients).all()
            or type(gamma) not in (int,float) or not math.isfinite(gamma) or gamma <= 0
            or type(intercept) not in (int,float) or not math.isfinite(intercept)):
        raise ValueError('Invalid contextual kernel classifier')
    result = []
    norms = np.sum(vectors*vectors, axis=1)
    # Bounded batches avoid a query-by-support matrix for the entire population.
    for start in range(0, len(values), 128):
        chunk = values[start:start+128]
        distance = np.maximum(np.sum(chunk*chunk,axis=1)[:,None]+norms[None,:]-2*(chunk@vectors.T),0.)
        result.extend((np.exp(-gamma*distance)@coefficients+intercept).tolist())
    return np.asarray(result)


def scores(features, head, support):
    import numpy as np
    values = margins(features,head,support)
    # Monotone bounded margin, explicitly not a calibrated probability.
    return .5 + np.arctan(values)/np.pi
