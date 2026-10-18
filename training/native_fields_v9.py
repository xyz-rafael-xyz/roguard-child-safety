"""Factorized interpretation heads over a fixed native encoder representation.

Policy mass is exact for the stated independent-field distribution, not a
calibrated estimate of safety or a learned joint distribution.
"""
import itertools
from roguard.evidence import ONTOLOGY, disposition

FIELDS=tuple(ONTOLOGY)
SIZES=tuple(len(ONTOLOGY[f]) for f in FIELDS)
WORLD_INDICES=tuple(itertools.product(*(range(n) for n in SIZES)))
WORLD_ACTIONS=tuple(int(disposition({f:{'label':ONTOLOGY[f][v]} for f,v in zip(FIELDS,world)})[0]!='CONTINUE')
                    for world in WORLD_INDICES)


def policy_mass(probabilities):
    """Batch independent categorical marginals, with explicit numerical checks."""
    import numpy as np
    if len(probabilities)!=len(FIELDS):raise ValueError('Six field distributions required')
    values=[np.asarray(p,dtype=np.float64) for p in probabilities]
    rows=values[0].shape[0] if values[0].ndim==2 else 0
    if not rows:raise ValueError('Empty policy input')
    for p,size in zip(values,SIZES):
        if (p.shape!=(rows,size) or not np.isfinite(p).all() or (p<0).any() or (p>1).any()
                or not np.allclose(p.sum(axis=1),1,atol=1e-6,rtol=0)):
            raise ValueError('Invalid field probabilities')
    result=np.zeros(rows,dtype=np.float64)
    for world,action in zip(WORLD_INDICES,WORLD_ACTIONS):
        if action:
            mass=np.ones(rows,dtype=np.float64)
            for p,index in zip(values,world):mass*=p[:,index]
            result+=mass
    if not np.isfinite(result).all() or (result < -1e-6).any() or (result > 1+1e-6).any():
        raise ValueError('Invalid integrated policy mass')
    return np.clip(result,0,1)


def targets(rows):
    return [[ONTOLOGY[f].index(row['attributes'][f]) for f in FIELDS] for row in rows]


def network(torch,dimensions,hidden):
    if hidden==0:return torch.nn.Linear(dimensions,sum(SIZES))
    return torch.nn.Sequential(torch.nn.Linear(dimensions,hidden),torch.nn.GELU(),
                               torch.nn.Dropout(.1),torch.nn.Linear(hidden,sum(SIZES)))


def probabilities(torch,logits):
    return [torch.softmax(value,dim=-1).cpu().numpy() for value in torch.split(logits,SIZES,dim=-1)]
