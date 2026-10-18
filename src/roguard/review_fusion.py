"""Explicit experimental fusion of input-bound native and interpreter reports.

The fitted component does not load neural models, choose a default model,
authenticate upstream reports or authorize external actions.
"""
import hashlib
import json
import math
from pathlib import Path

from .evidence import ONTOLOGY, disposition
from .interpreter import CONTRACT_DIGEST
from .native_review import prepare_input

FIELDS = tuple(ONTOLOGY)
FEATURE_CONTRACT = dict(fields=FIELDS, ontology=ONTOLOGY,
    features='native score, logit clipped to probability[1e-6,1-1e-6], conditional probabilities, predicted-label one-hot, native score times probabilities',
    preprocessing='per-feature fitted mean and scale; intercept; logistic logit clamped[-700,700]',
    invalid='Any component abstention remains an abstention')
FEATURE_DIGEST = hashlib.sha256(json.dumps(FEATURE_CONTRACT, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
DIMENSION = 2+3*sum(len(v) for v in ONTOLOGY.values())


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a lowercase SHA-256 digest')
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def feature_vector(score, structured):
    if not _number(score) or not 0 <= score <= 1:
        raise ValueError('Invalid native score')
    probabilities, labels = [], []
    values = structured.get('interpretation')
    vectors = structured.get('conditional_field_probabilities')
    if (not isinstance(values, dict) or set(values) != set(FIELDS)
            or not isinstance(vectors, dict) or set(vectors) != set(FIELDS)):
        raise ValueError('Incomplete interpreter fields')
    for field in FIELDS:
        p = vectors[field]
        if (not isinstance(p, list) or len(p) != len(ONTOLOGY[field])
                or any(not _number(v) or not 0 <= v <= 1 for v in p) or abs(sum(p)-1) > 1e-5):
            raise ValueError('Invalid interpreter probabilities')
        label = values[field]
        if label not in ONTOLOGY[field] or ONTOLOGY[field][max(range(len(p)), key=p.__getitem__)] != label:
            raise ValueError('Interpreter label differs from probability argmax')
        probabilities.extend(p)
        labels.extend(float(label == item) for item in ONTOLOGY[field])
    bounded = min(1-1e-6, max(1e-6, score))
    return [score, math.log(bounded/(1-bounded))]+probabilities+labels+[score*p for p in probabilities]


class ReviewFusion:
    def __init__(self, artifact, *, expected_artifact_sha256):
        with Path(artifact).open('rb') as stream:
            raw = stream.read(65537)
        if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != _digest(expected_artifact_sha256):
            raise ValueError('Fusion artifact size or identity differs')
        spec = json.loads(raw)
        required = {'format','language','native_artifact_sha256','interpreter_archive_sha256','interpreter_contract_digest',
                    'feature_contract_digest','selection_sha256','model','threshold'}
        if not isinstance(spec, dict) or set(spec) != required or spec['format'] != 'roguard-review-fusion-experiment-v1':
            raise ValueError('Unsupported fusion artifact')
        if spec['language'] not in ('ro','uk') or spec['interpreter_contract_digest'] != CONTRACT_DIGEST or spec['feature_contract_digest'] != FEATURE_DIGEST:
            raise ValueError('Fusion language or feature contract differs')
        for field in ('native_artifact_sha256','interpreter_archive_sha256','selection_sha256'):
            _digest(spec[field])
        model = spec['model']
        if not isinstance(model, dict) or set(model) != {'mean','scale','coefficients'}:
            raise ValueError('Invalid fusion coefficient schema')
        for name, dimension in (('mean',DIMENSION),('scale',DIMENSION),('coefficients',DIMENSION+1)):
            values = model[name]
            if not isinstance(values, list) or len(values) != dimension or any(not _number(v) for v in values):
                raise ValueError('Invalid fusion coefficient values')
        if any(v <= 0 for v in model['scale']) or not _number(spec['threshold']) or not 0 <= spec['threshold'] <= 1:
            raise ValueError('Invalid fusion scaling or threshold')
        self._spec = spec
        self.identity = expected_artifact_sha256

    def predict(self, payload, native, interpreter, *, focus=None):
        _, focus, context = prepare_input(payload, focus)
        spec = self._spec
        if payload['language'] != spec['language']:
            raise ValueError('Conversation language differs from fusion model')
        for report, format_name, expected in ((native,'roguard-native-review-v1',spec['native_artifact_sha256']),
                                              (interpreter,'roguard-interpreter-v1',spec['interpreter_archive_sha256'])):
            if (not isinstance(report, dict) or report.get('format') != format_name or report.get('model_identity') != expected
                    or report.get('language') != spec['language'] or report.get('input_digest') != context
                    or type(report.get('focus')) is not int or report['focus'] != focus):
                raise ValueError('Component model or input binding differs')
            for flag in ('text_in_report','establishes_safety','permits_external_action','permits_notification'):
                if report.get(flag) is not False:
                    raise ValueError('Unsupported component claim')
        if (interpreter.get('contract_digest') != CONTRACT_DIGEST or interpreter.get('probabilities_are_calibrated') is not False
                or native.get('score_is_calibrated') is not False or type(interpreter.get('valid_label_tokens')) is not bool):
            raise ValueError('Component interpretation contract differs')
        native_score = native.get('score')
        if native_score is not None and (not _number(native_score) or not 0 <= native_score <= 1):
            raise ValueError('Invalid native score')
        if native_score is not None and (type(native.get('predicted')) is not int or native['predicted'] not in (0,1)):
            raise ValueError('Invalid native proposal')
        if (native_score is None and native.get('predicted') is not None) or (not interpreter['valid_label_tokens'] and interpreter.get('predicted') is not None):
            raise ValueError('Invalid component must abstain')
        abstention = native_score is None or not interpreter['valid_label_tokens']
        score = None
        if not abstention:
            vector = feature_vector(native_score, interpreter)
            proposal = int(disposition({k: {'label': v} for k,v in interpreter['interpretation'].items()})[0] != 'CONTINUE')
            if type(interpreter.get('predicted')) is not int or interpreter['predicted'] != proposal:
                raise ValueError('Interpreter policy proposal differs')
            model = spec['model']
            logit = model['coefficients'][0]+sum(w*(x-m)/s for w,x,m,s in zip(model['coefficients'][1:],vector,model['mean'],model['scale']))
            if not math.isfinite(logit):
                raise ValueError('Nonfinite fusion logit')
            score = 1/(1+math.exp(-max(-700,min(700,logit))))
        predicted = None if score is None else int(score >= spec['threshold'])
        return dict(format='roguard-review-fusion-v1', language=spec['language'], focus=focus, input_digest=context,
                    model_identity=self.identity, native_model_identity=spec['native_artifact_sha256'],
                    interpreter_model_identity=spec['interpreter_archive_sha256'], feature_contract_digest=FEATURE_DIGEST,
                    predicted=predicted, score=score, decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',
                    reason='COMPONENT_ABSTENTION' if predicted is None else 'FUSION_REVIEW_PROPOSAL',
                    review_requested=predicted != 0, score_is_calibrated=False, reports_authenticated=False,
                    text_in_report=False, establishes_safety=False, permits_external_action=False, permits_notification=False)
