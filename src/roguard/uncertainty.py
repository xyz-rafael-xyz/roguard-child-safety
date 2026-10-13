"""Model-bound class-conditional split conformal sets for binary scores."""
import hashlib,json,math


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def probability(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('Expected a finite score in [0,1]')
    return float(value)


class CalibratedDecisions:
    """Coverage requires exchangeable calibration/test examples per class.

    Does not estimate individual safety probabilities or guarantee under shift.
    Model identity must include preprocessing, weights and the scoring task.
    """
    def __init__(self, artifact):
        a=json.loads(json.dumps(artifact,allow_nan=False))
        if set(a)!={'schema','model_id','language','alpha','thresholds','counts','calibration_digest'} or a['schema']!='roguard-conformal-v1':
            raise ValueError('Invalid calibration artifact')
        if a['language'] not in ('ro','uk') or not isinstance(a['model_id'],str) or not a['model_id']:
            raise ValueError('Invalid calibration identity')
        if not 0<probability(a['alpha'])<1:raise ValueError('Invalid alpha')
        if set(a['counts'])!={'0','1'} or set(a['thresholds'])!={'0','1'}:raise ValueError('Both classes required')
        for key in ('0','1'):
            if type(a['counts'][key]) is not int or a['counts'][key]<1:raise ValueError('Empty class')
            probability(a['thresholds'][key])
        self.artifact=a

    @classmethod
    def fit(cls, scores, labels, *, model_id, language, alpha=.1):
        probability(alpha)
        if not 0<alpha<1 or not 2<=len(scores)<=100000 or len(scores)!=len(labels):raise ValueError('Invalid calibration sample')
        scores=[probability(s) for s in scores]
        if any(type(y) is not int or y not in (0,1) for y in labels):raise ValueError('Binary labels required')
        thresholds={};counts={}
        for y in (0,1):
            residuals=sorted((1-s if y else s) for s,t in zip(scores,labels) if t==y)
            n=len(residuals)
            if not n:raise ValueError('Both classes need calibration examples')
            rank=math.ceil((n+1)*(1-alpha))
            thresholds[str(y)]=residuals[rank-1] if rank<=n else 1.0;counts[str(y)]=n
        return cls(dict(schema='roguard-conformal-v1',model_id=model_id,language=language,alpha=alpha,thresholds=thresholds,counts=counts,calibration_digest=fingerprint(list(zip(scores,labels)))))

    def predict(self, score, *, model_id, language):
        if model_id!=self.artifact['model_id'] or language!=self.artifact['language']:
            raise ValueError('Calibration does not match model, task or language')
        s=probability(score); t=self.artifact['thresholds']
        labels=[y for y in (0,1) if (1-s if y else s)<=t[str(y)]]
        return {'prediction_set':labels,'decision':('NO_MODEL_SIGNAL' if labels==[0] else 'REVIEW_SUPPORT_SIGNAL' if labels==[1] else 'INSUFFICIENT_EVIDENCE'),
                'alpha':self.artifact['alpha'],'calibration_counts':self.artifact['counts'],
                'coverage_requires_exchangeability':True,'real_world_coverage_established':False}
