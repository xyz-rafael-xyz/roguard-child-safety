"""Factorized interpretation heads over a fixed native encoder representation.

Policy mass is exact for the declared factor distribution, with optional
training-fitted pair potentials. It is not a calibrated estimate of safety.
"""
import itertools
from roguard.evidence import ONTOLOGY, disposition

FIELDS=tuple(ONTOLOGY)
SIZES=tuple(len(ONTOLOGY[f]) for f in FIELDS)
WORLD_INDICES=tuple(itertools.product(*(range(n) for n in SIZES)))
WORLD_ACTIONS=tuple(int(disposition({f:{'label':ONTOLOGY[f][v]} for f,v in zip(FIELDS,world)})[0]!='CONTINUE')
                    for world in WORLD_INDICES)


def policy_mass(probabilities,couplings=None):
    """Batch head marginals with optional pair potentials and numerical checks."""
    import numpy as np
    if len(probabilities)!=len(FIELDS):raise ValueError('Six field distributions required')
    values=[np.asarray(p,dtype=np.float64) for p in probabilities]
    rows=values[0].shape[0] if values[0].ndim==2 else 0
    if not rows:raise ValueError('Empty policy input')
    for p,size in zip(values,SIZES):
        if (p.shape!=(rows,size) or not np.isfinite(p).all() or (p<0).any() or (p>1).any()
                or not np.allclose(p.sum(axis=1),1,atol=1e-6,rtol=0)):
            raise ValueError('Invalid field probabilities')
    def p(field,label):return values[FIELDS.index(field)][:,ONTOLOGY[field].index(label)]
    if couplings is not None:
        pairs={}
        for name,left,right in (('scope','subject','source'),('assertion_relation','assertion','relation')):
            weights=np.asarray(couplings[name],dtype=np.float64)
            if (weights.shape!=(len(ONTOLOGY[left]),len(ONTOLOGY[right]))
                    or not np.isfinite(weights).all() or (weights<=0).any()):raise ValueError('Invalid coupling table')
            joint=values[FIELDS.index(left)][:,:,None]*values[FIELDS.index(right)][:,None,:]*weights[None,:,:]
            pairs[name]=joint/joint.sum(axis=(1,2))[:,None,None]
        scope=pairs['scope'];ar=pairs['assertion_relation']
        sf=ONTOLOGY['subject'].index('FICTIONAL');fi=ONTOLOGY['source'].index('FICTION')
        neg=ONTOLOGY['assertion'].index('NEGATED');ret=ONTOLOGY['assertion'].index('RETRACTED');new=ONTOLOGY['relation'].index('NEW')
        mismatch=sum(scope[:,i,j] for i in range(scope.shape[1]) for j in range(scope.shape[2]) if (i==sf)!=(j==fi))
        real=sum(scope[:,i,j] for i in range(scope.shape[1]) for j in range(scope.shape[2]) if i!=sf and j!=fi)
        fiction_review=1-sum(ar[:,i,new] for i in range(ar.shape[1]) if i!=ret)
        result=p('concern','1')*(mismatch+scope[:,sf,fi]*fiction_review+real*(1-ar[:,neg,new]))
        if not np.isfinite(result).all():raise ValueError('Invalid coupled policy mass')
        return np.clip(result,0,1)
    fictional=p('subject','FICTIONAL');fiction=p('source','FICTION')
    negated=p('assertion','NEGATED');retracted=p('assertion','RETRACTED');retracts=p('relation','RETRACTS')
    mismatch=fictional*(1-fiction)+(1-fictional)*fiction
    fiction_review=retracted+retracts-retracted*retracts
    real_review=1-(1-retracts)*negated
    result=p('concern','1')*(mismatch+fictional*fiction*fiction_review+(1-fictional)*(1-fiction)*real_review)
    if not np.isfinite(result).all() or (result < -1e-6).any() or (result > 1+1e-6).any():
        raise ValueError('Invalid integrated policy mass')
    return np.clip(result,0,1)


def fit_couplings(rows,weights):
    """Training-only pair likelihood ratios, with one pseudocount per cell."""
    import numpy as np
    if len(rows)!=len(weights) or any(not np.isfinite(v) or v<=0 for v in weights):raise ValueError('Invalid coupling weights')
    result={}
    for name,left,right in (('scope','subject','source'),('assertion_relation','assertion','relation')):
        table=np.ones((len(ONTOLOGY[left]),len(ONTOLOGY[right])),dtype=np.float64)
        for row,weight in zip(rows,weights):
            table[ONTOLOGY[left].index(row['attributes'][left]),ONTOLOGY[right].index(row['attributes'][right])]+=weight
        joint=table/table.sum();factor=joint/(joint.sum(axis=1)[:,None]*joint.sum(axis=0)[None,:])
        result[name]=factor.tolist()
    return result


def targets(rows):
    return [[ONTOLOGY[f].index(row['attributes'][f]) for f in FIELDS] for row in rows]


def network(torch,dimensions,hidden):
    if hidden==0:return torch.nn.Linear(dimensions,sum(SIZES))
    return torch.nn.Sequential(torch.nn.Linear(dimensions,hidden),torch.nn.GELU(),
                               torch.nn.Dropout(.1),torch.nn.Linear(hidden,sum(SIZES)))


def probabilities(torch,logits):
    return [torch.softmax(value,dim=-1).cpu().numpy() for value in torch.split(logits,SIZES,dim=-1)]
