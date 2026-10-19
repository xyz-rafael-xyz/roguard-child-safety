"""Preserve base reviews and add bounded, scope-consistent model agreement."""
from .assurance import policy_digest
from .evidence import ONTOLOGY
from .language_bundle import _read_manifest
from .review_fusion import _digest, _number
import hashlib

CONTRACT_DIGEST = hashlib.sha256(b'scope-consensus-real-active-max-agreed-scope-min-concern-margin-v1').hexdigest()


class ScopeConsensusReview:
    def __init__(self, artifact, base, *, expected_sha256):
        from pathlib import Path
        spec = _read_manifest(Path(artifact), expected_sha256)
        keys = {'format','base_artifact_sha256','native_threshold','base_threshold',
                'rescue_threshold','selection_sha256','policy_digest','contract_digest'}
        if (not isinstance(spec,dict) or set(spec)!=keys
                or spec['format']!='roguard-scope-consensus-artifact-v1'
                or spec['contract_digest']!=CONTRACT_DIGEST or spec['policy_digest']!=policy_digest()):
            raise ValueError('Unsupported scope consensus artifact')
        for key in ('base_artifact_sha256','selection_sha256'):_digest(spec[key])
        if (spec['base_artifact_sha256']!=base.identity
                or spec['base_threshold']!=base._spec['threshold']
                or any(not _number(spec[k]) or not 0<=spec[k]<=1 for k in ('native_threshold','base_threshold'))
                or not _number(spec['rescue_threshold']) or not .5<=spec['rescue_threshold']<=1.000001):
            raise ValueError('Consensus component identity or threshold differs')
        self.config=spec;self.base=base;self._spec=base._spec;self.identity=expected_sha256

    def predict(self,payload,native,semantic,interpreter,*,focus=None):
        # The existing combiner validates every input/model binding, probability
        # vector, interpretation and permission flag before any rescue can occur.
        original=self.base.predict(payload,native,semantic,interpreter,focus=focus)
        score=original['score'];confidence=None
        if score is not None:
            native_positive=int(native['score']>=self.config['native_threshold'])
            if native['predicted']!=native_positive:
                raise ValueError('Native decision differs from pinned consensus cutoff')
            labels=interpreter['interpretation'];other=semantic['interpretation']
            eligible=native_positive and interpreter['predicted']==1 and all(
                v['subject']!='FICTIONAL' and v['source']!='FICTION' and v['assertion']!='NEGATED'
                for v in (labels,other))
            if eligible:
                p=interpreter['conditional_field_probabilities'];q=semantic['field_probabilities']
                def value(values,field,label):return values[field][ONTOLOGY[field].index(label)]
                confidence=min(value(p,'concern','1'),
                    max(1-value(p,'subject','FICTIONAL'),1-value(q,'subject','FICTIONAL')),
                    max(1-value(p,'source','FICTION'),1-value(q,'source','FICTION')),
                    max(1-value(p,'assertion','NEGATED'),1-value(q,'assertion','NEGATED')))
            margin=score-self.config['base_threshold']
            if confidence is not None:margin=max(margin,confidence-self.config['rescue_threshold'])
            score=.5+.5*margin
        predicted=None if score is None else int(score>=.5)
        return dict(original,format='roguard-scope-consensus-review-v1',model_identity=self.identity,
            base_model_identity=self.base.identity,contract_digest=CONTRACT_DIGEST,
            base_score=original['score'],score=score,predicted=predicted,
            decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',
            reason='COMPONENT_ABSTENTION' if predicted is None else 'SCOPE_CONSENSUS_REVIEW_PROPOSAL',
            review_requested=predicted!=0,score_is_calibrated=False,score_kind='normalized_decision_margin',
            rescue_confidence=confidence,rescue_applied=original['predicted']==0 and predicted==1)
