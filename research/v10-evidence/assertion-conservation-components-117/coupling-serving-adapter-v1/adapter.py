"""Stage0 float32-logit guard around the frozen assertion KL projection.

No model, encoder or training. Accepted outputs preserve the other15 logits
bitwise; any rejected reconstruction returns all19 parent logits bitwise.
The original1e-12 score tolerance is retained, never tuned for acceptance rate.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import numpy as np
from roguard.gemma4_review import _score

PROJECTION_PATH=Path(__file__).resolve().parent.parent/'coupling-projection-stage0-v2/projection.py'
PROJECTION_SHA256='aec96156c0c029c56c490c89802bc22b57e7c3cbe8cbd06b19e11276e99a2959'
if hashlib.sha256(PROJECTION_PATH.read_bytes()).hexdigest()!=PROJECTION_SHA256:
    raise ValueError('Frozen probability projection source differs')
_spec=importlib.util.spec_from_file_location('frozen_stage0_probability_projection',PROJECTION_PATH)
projection=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(projection)
ASSERTION=slice(6,10)
OTHER=np.array([0,1,2,3,4,5,10,11,12,13,14,15,16,17,18])


def field_probabilities(logits):
    """Match portable _score float64 softmax operation order exactly."""
    values=np.asarray(logits,dtype=np.float64)
    fields={}
    for name,part in zip(projection.FIELD_SIZES,np.split(values[:17],[3,6,10,13,15])):
        p=np.exp(part-part.max());fields[name]=(p/p.sum()).tolist()
    return fields


def adapt(parent_logits,spec,*,residual=None,target_assertion=None,score_tolerance=1e-12):
    if (not isinstance(parent_logits,np.ndarray) or parent_logits.dtype!=np.float32
            or parent_logits.shape!=(19,) or not np.isfinite(parent_logits).all()):
        raise ValueError('Expected finite original19 float32 logits')
    if spec.get('mode')!='coupled_policy' or spec.get('couplings') is None:
        raise ValueError('This component requires the fixed coupled-policy scorer')
    if (type(score_tolerance) not in (int,float) or not math.isfinite(score_tolerance)
            or not 0<score_tolerance<=1e-12):
        raise ValueError('Cannot relax the original stage0 score tolerance')
    if (residual is None)==(target_assertion is None):
        raise ValueError('Supply exactly one residual or target assertion')
    original=parent_logits.copy();fields=field_probabilities(original)
    target=(projection.conditional_au_target(fields['assertion'],residual)
            if residual is not None else target_assertion)
    parent_score=_score(original,spec)
    probability=projection.project_assertion(fields,target,spec['couplings'],
        threshold=spec['threshold'],risk_tolerance=score_tolerance)
    if probability['parent_score']!=parent_score:
        raise ValueError('Probability path differs from actual parent serving score')
    attempted_score=None
    attempted_sign=None
    attempted_probability_delta=None

    def result(logits,status,reason=None):
        actual=_score(logits,spec)
        if (abs(actual-parent_score)>score_tolerance
                or (actual>=spec['threshold'])!=(parent_score>=spec['threshold'])
                or logits[OTHER].tobytes()!=original[OTHER].tobytes()):
            raise ValueError('Final serving invariant failed')
        if reason is not None and logits.tobytes()!=original.tobytes():
            raise ValueError('Fallback must retain exact parent logits')
        return dict(logits=logits,status=status,fallback_reason=reason,
            parent_score=parent_score,output_score=actual,score_difference=actual-parent_score,
            binary_decision_unchanged=(actual>=spec['threshold'])==(parent_score>=spec['threshold']),
            other15_logits_bitwise_unchanged=True,parent19_bitwise_unchanged=logits.tobytes()==original.tobytes(),
            attempted_score=attempted_score,
            attempted_score_difference=None if attempted_score is None else attempted_score-parent_score,
            attempted_binary_decision_unchanged=attempted_sign,
            attempted_maximum_probability_difference=attempted_probability_delta,
            probability_projection_status=probability['status'],
            probability_projection_fallback_reason=probability['fallback_reason'],
            probability_projection_score_difference=probability['score_difference'],
            projection_source_sha256=PROJECTION_SHA256,score_tolerance=score_tolerance,qualified=False)

    if probability['status']=='IDENTITY_TARGET':
        return result(original,'EXACT_PARENT_IDENTITY')
    if probability['status']=='IDENTITY_FALLBACK':
        return result(original,'EXACT_PARENT_FALLBACK','probability_projection:'+probability['fallback_reason'])
    q=np.asarray(probability['fields']['assertion'],dtype=np.float64)
    if not np.isfinite(q).all() or np.any(q<=0):
        return result(original,'EXACT_PARENT_FALLBACK','zero_or_nonfinite_projected_probability')
    candidate=original.copy()
    candidate[ASSERTION]=np.log(q).astype(np.float32)
    if not np.isfinite(candidate).all():
        return result(original,'EXACT_PARENT_FALLBACK','nonfinite_float32_reconstruction')
    attempted_score=_score(candidate,spec)
    attempted_sign=(attempted_score>=spec['threshold'])==(parent_score>=spec['threshold'])
    roundtrip=field_probabilities(candidate)['assertion']
    attempted_probability_delta=float(np.max(np.abs(np.asarray(roundtrip)-q)))
    if not attempted_sign:
        return result(original,'EXACT_PARENT_FALLBACK','float32_threshold_crossing')
    if abs(attempted_score-parent_score)>score_tolerance:
        return result(original,'EXACT_PARENT_FALLBACK','float32_score_tolerance')
    return result(candidate,'FLOAT32_GUARDED_PROJECTION')
