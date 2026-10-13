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

class FeatureSupport:
    """Diagonal-distance diagnostic, not a semantic novelty detector or guarantee."""
    def __init__(self,artifact):
        self.artifact=json.loads(json.dumps(artifact,allow_nan=False))
        a=self.artifact
        if set(a)!={'model_id','center','scale','limit','dimension','reference_count','calibration_count'}:raise ValueError('Invalid support artifact')
        if not isinstance(a['model_id'],str) or type(a['dimension']) is not int or not 1<=a['dimension']<=4096 or len(a['center'])!=a['dimension'] or len(a['scale'])!=a['dimension']:raise ValueError('Invalid feature dimension')
        if any(type(x) not in (float,int) or not math.isfinite(x) for x in a['center']+a['scale']+[a['limit']]) or min(a['scale'])<=0 or a['limit']<0:raise ValueError('Invalid feature support')
    @classmethod
    def fit(cls,reference,calibration,*,model_id):
        if not 2<=len(reference)<=10000 or not 2<=len(calibration)<=10000:raise ValueError('Insufficient or oversized feature sample')
        d=len(reference[0])
        if not 1<=d<=4096 or any(len(row)!=d or any(type(v) not in (float,int) or not math.isfinite(v) for v in row) for row in reference+calibration):raise ValueError('Invalid features')
        center=[sum(row[i] for row in reference)/len(reference) for i in range(d)]
        scale=[max(1e-6,math.sqrt(sum((row[i]-center[i])**2 for row in reference)/len(reference))) for i in range(d)]
        distances=sorted(sum(((v-c)/s)**2 for v,c,s in zip(row,center,scale))/d for row in calibration)
        return cls(dict(model_id=model_id,center=center,scale=scale,limit=distances[math.ceil(.95*len(distances))-1],dimension=d,reference_count=len(reference),calibration_count=len(calibration)))
    def check(self,features,*,model_id):
        a=self.artifact
        if model_id!=a['model_id'] or len(features)!=a['dimension'] or any(type(v) not in (float,int) or not math.isfinite(v) for v in features):raise ValueError('Incompatible features')
        distance=sum(((v-c)/s)**2 for v,c,s in zip(features,a['center'],a['scale']))/a['dimension']
        return dict(distance=distance,outside_reference_support=distance>a['limit'],semantic_novelty_verified=False)
