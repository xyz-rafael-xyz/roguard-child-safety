"""Joint bilingual semantic heads over frozen Gemma4 contextual features."""
from training.native_fields_v9 import FIELDS,SIZES


def network(torch,dimensions,hidden=128):
    return torch.nn.Sequential(torch.nn.Linear(dimensions,hidden),torch.nn.GELU(),
        torch.nn.Dropout(.1),torch.nn.Linear(hidden,sum(SIZES)+2))


def portable_logits(features,tensors):
    """Independent numeric export path; no executable model serialization."""
    import math
    import numpy as np
    from training.gemma4_readout_v9 import normalize
    values=normalize(features).astype(np.float32)
    first=values@tensors['0.weight'].T+tensors['0.bias']
    erf=np.vectorize(math.erf,otypes=[np.float64])(first.astype(np.float64)/math.sqrt(2.))
    hidden=(first*.5*(1.+erf)).astype(np.float32)
    return hidden@tensors['3.weight'].T+tensors['3.bias']
