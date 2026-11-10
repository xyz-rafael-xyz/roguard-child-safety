"""Synthetic stage0 only: 9x2 transportation bounds and frozen-policy parity.

No dataset, tokenizer, model, fitting, admission or production integration.
Exact rational bounds are mathematically exact for supplied rational inputs.
Float bounds include an explicit non-certified roundoff guard; boundaries remain
unresolved, never silently promoted to exact decision guarantees.
"""
from fractions import Fraction
from itertools import product
import math

SUBJECTS=('SELF','OTHER','FICTIONAL')
SOURCES=('DIRECT','REPORTED','FICTION')
ASSERTIONS=('AFFIRMED','NEGATED','UNCERTAIN','RETRACTED')
RELATIONS=('NEW','RETRACTS')
TIMES=('CURRENT','PAST','HYPOTHETICAL')
SCOPES=tuple(product(SUBJECTS,SOURCES))
TOLERANCE=1e-12


def require(ok,message):
 if not ok:raise ValueError(message)


def _exact(value):
 require(type(value) is int or isinstance(value,Fraction),'Exact API requires Fraction/int; floats need explicit conversion')
 return Fraction(value)


def _finite(value):
 require(type(value) in (int,float) and math.isfinite(value),'Expected finite real scalar')
 return float(value)


def _check_inputs(rows,positive_mass,weights,exact=False):
 cast=_exact if exact else _finite
 rows=tuple(map(cast,rows));p=cast(positive_mass);weights=tuple(map(cast,weights))
 require(len(rows)==len(weights)==9,'Expected all nine scope cells')
 require(all(0<=x<=1 for x in rows+weights) and 0<=p<=1,'Invalid marginal/cost probability')
 total=sum(rows) if exact else math.fsum(rows)
 require(total==1 if exact else abs(total-1)<=TOLERANCE,'Row probabilities do not sum to one')
 return rows,p,weights


def _allocation(rows,p,weights,descending):
 remaining=p;positive=[p*0 for _ in rows]
 for index in sorted(range(9),key=lambda i:((-weights[i] if descending else weights[i]),i)):
  take=min(rows[index],remaining);positive[index]=take;remaining-=take
 return tuple(positive),remaining


def greedy_bounds_exact(rows,positive_mass,weights):
 """Fractional-knapsack optimum; all 9x2 row/column marginals preserved exactly."""
 rows,p,weights=_check_inputs(rows,positive_mass,weights,True)
 lo,left=_allocation(rows,p,weights,False);hi,right=_allocation(rows,p,weights,True)
 require(left==right==0,'Impossible rational capacity allocation')
 score=lambda xs:sum((x*w for x,w in zip(xs,weights)),Fraction(0))
 minimum,maximum=score(lo),score(hi)
 require(0<=minimum<=maximum<=p,'Exact concern bound violated')
 table=lambda xs:tuple((r-x,x) for r,x in zip(rows,xs))
 return dict(minimum=minimum,maximum=maximum,min_positive=lo,max_positive=hi,
             min_table=table(lo),max_table=table(hi),positive_mass=p,rows=rows,weights=weights)


def reachability_exact(bounds,threshold):
 threshold=_exact(threshold);require(0<=threshold<=1,'Invalid threshold')
 return dict(positive_feasible=bounds['maximum']>=threshold,
             negative_feasible=bounds['minimum']<threshold,
             positive_impossible_by_concern=bounds['positive_mass']<threshold)


def greedy_bounds(rows,positive_mass,weights):
 """Float64 estimates plus witnesses/explicit guard, not certified interval bounds."""
 rows,p,weights=_check_inputs(rows,positive_mass,weights)
 lo,left=_allocation(rows,p,weights,False);hi,right=_allocation(rows,p,weights,True)
 require(max(abs(left),abs(right))<=TOLERANCE,'Numerically infeasible capacity allocation')
 score=lambda xs:math.fsum(x*w for x,w in zip(xs,weights))
 minimum,maximum=score(lo),score(hi)
 table=lambda xs:tuple((r-x,x) for r,x in zip(rows,xs))
 low_table,high_table=table(lo),table(hi)
 residual=max(abs(math.fsum(rows)-1),abs(math.fsum(lo)-p),abs(math.fsum(hi)-p),abs(left),abs(right),
              *(abs(math.fsum(cell)-r) for tab in (low_table,high_table) for cell,r in zip(tab,rows)))
 guard=max(1e-12,64*math.ulp(1.0)+8*residual)
 require(-guard<=minimum<=maximum+guard and maximum<=p+guard,'Concern bound exceeded beyond numerical guard')
 interval=lambda value:(max(0.,value-guard),min(p,value+guard))
 return dict(minimum=minimum,maximum=maximum,min_positive=lo,max_positive=hi,
             min_table=low_table,max_table=high_table,positive_mass=p,rows=rows,weights=weights,
             minimum_guard_interval=interval(minimum),maximum_guard_interval=interval(maximum),
             guard=guard,maximum_marginal_roundoff=residual,
             numerical_contract='Explicit diagnostic guard only; not certified floating-point intervals. Use exact rational oracle for exact equality witnesses.')


def reachability(bounds,threshold):
 """Served positive uses >=; served negative uses strict <. Boundary stays unknown."""
 threshold=_finite(threshold);require(0<=threshold<=1,'Invalid threshold')
 p=bounds['positive_mass']
 if p<threshold:
  return dict(positive='IMPOSSIBLE_CONCERN_CAP',negative='FEASIBLE_CONCERN_CAP',
              positive_impossible_by_concern=True,numerical_boundary=False)
 lo_min,hi_min=bounds['minimum_guard_interval'];lo_max,hi_max=bounds['maximum_guard_interval']
 positive=('FEASIBLE_OUTSIDE_GUARD' if lo_max>=threshold else
           'IMPOSSIBLE_OUTSIDE_GUARD' if hi_max<threshold else 'NUMERICAL_BOUNDARY_UNRESOLVED')
 negative=('FEASIBLE_OUTSIDE_GUARD' if hi_min<threshold else
           'IMPOSSIBLE_OUTSIDE_GUARD' if lo_min>=threshold else 'NUMERICAL_BOUNDARY_UNRESOLVED')
 return dict(positive=positive,negative=negative,positive_impossible_by_concern=False,
             numerical_boundary='NUMERICAL_BOUNDARY_UNRESOLVED' in (positive,negative))


def _vector(values,n):
 values=tuple(map(_finite,values));require(len(values)==n and all(0<=x<=1 for x in values),'Invalid probability vector')
 require(abs(math.fsum(values)-1)<=TOLERANCE,'Probability vector not normalized')
 return values


def parent_distributions(values,couplings=None):
 """Independently construct the two coupled pairs; caller field vectors unchanged."""
 sizes=dict(subject=3,time=3,assertion=4,source=3,relation=2,concern=2)
 require(set(values)==set(sizes),'Field inventory differs')
 q={k:_vector(values[k],n) for k,n in sizes.items()}
 def pair(left,right,name):
  factors=couplings[name] if couplings is not None else [[1.]*len(right) for _ in left]
  require(len(factors)==len(left) and all(len(row)==len(right) for row in factors),'Coupling shape differs')
  c=[[_finite(v) for v in row] for row in factors];require(all(v>0 for row in c for v in row),'Couplings must be positive')
  raw=[[x*y*c[i][j] for j,y in enumerate(right)] for i,x in enumerate(left)]
  total=math.fsum(v for row in raw for v in row);require(total>0,'Coupled distribution has no mass')
  return tuple(tuple(v/total for v in row) for row in raw)
 if couplings is not None:require(set(couplings)=={'scope','assertion_relation'},'Coupling inventory differs')
 scope=pair(q['subject'],q['source'],'scope');ar=pair(q['assertion'],q['relation'],'assertion_relation')
 return dict(scope=tuple(v for row in scope for v in row),assertion_relation=ar,concern=q['concern'],time=q['time'])


def policy_weights(assertion_relation,policy):
 """Enumerate the existing policy to avoid encoding new semantic risk rules."""
 ar=tuple(tuple(map(_finite,row)) for row in assertion_relation)
 require(len(ar)==4 and all(len(row)==2 for row in ar),'Expected4x2 assertion/relation distribution')
 ar_total=math.fsum(v for row in ar for v in row)
 require(all(0<=v<=1 for row in ar for v in row) and abs(ar_total-1)<=TOLERANCE,'Invalid assertion/relation probabilities')
 weights=[]
 for subject,source in SCOPES:
  terms=[]
  for ai,assertion in enumerate(ASSERTIONS):
   for ri,relation in enumerate(RELATIONS):
    outcomes=[]
    for time in TIMES:
     attrs=dict(subject=subject,time=time,assertion=assertion,source=source,relation=relation,concern='0')
     require(policy(attrs)=='CONTINUE','Concern0 is not always nonreview; model requires rederivation')
     attrs['concern']='1';outcomes.append(int(policy(attrs)!='CONTINUE'))
    require(len(set(outcomes))==1,'Time affects binary policy; model requires rederivation')
    terms.append(ar[ai][ri]*outcomes[0])
  # Normalize the already-near-unit AR law explicitly; do not clip costs.
  # For all-review mismatch strata, numerator==denominator gives exact1.
  weights.append(math.fsum(terms)/ar_total)
 return tuple(weights)


def joint_score(table,weights):
 require(len(table)==len(weights)==9 and all(len(row)==2 for row in table),'Expected9x2 table and9weights')
 cells=tuple(tuple(map(_finite,row)) for row in table);costs=tuple(map(_finite,weights))
 require(all(0<=v<=1 for row in cells for v in row) and abs(math.fsum(v for row in cells for v in row)-1)<=TOLERANCE,'Invalid joint distribution')
 require(all(0<=w<=1+TOLERANCE for w in costs),'Invalid policy weights')
 return math.fsum(cell[1]*w for cell,w in zip(cells,costs))


def enumeration_score(values,couplings,policy):
 """Full432-world expectation, separate from the production closed form."""
 d=parent_distributions(values,couplings);terms=[]
 for si,(subject,source) in enumerate(SCOPES):
  for ai,assertion in enumerate(ASSERTIONS):
   for ri,relation in enumerate(RELATIONS):
    for ti,time in enumerate(TIMES):
     for ci,concern in enumerate(('0','1')):
      attrs=dict(subject=subject,time=time,assertion=assertion,source=source,relation=relation,concern=concern)
      mass=d['scope'][si]*d['assertion_relation'][ai][ri]*d['time'][ti]*d['concern'][ci]
      terms.append(mass*int(policy(attrs)!='CONTINUE'))
 return math.fsum(terms)


def zero_residual_score(values,couplings,portable_scorer,policy):
 """Exact scalar identity fast path plus independent mathematical parity checks."""
 parent=portable_scorer(values,couplings=couplings)
 d=parent_distributions(values,couplings);weights=policy_weights(d['assertion_relation'],policy)
 table=tuple((r*d['concern'][0],r*d['concern'][1]) for r in d['scope'])
 joint=joint_score(table,weights);enumerated=enumeration_score(values,couplings,policy)
 require(max(abs(parent-joint),abs(parent-enumerated))<=TOLERANCE,'Policy enumeration/parent parity failed')
 return dict(score=parent,parent_score=parent,independent_joint_score=joint,independent_enumerated_score=enumerated,
             maximum_parity_error=max(abs(parent-joint),abs(parent-enumerated)),zero_residual_identity=True,
             table=table,scope=d['scope'],assertion_relation=d['assertion_relation'],weights=weights)
