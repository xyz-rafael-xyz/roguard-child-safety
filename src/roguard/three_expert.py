"""Explicit JSON-tree combination of three input-bound local model reports."""
import hashlib
import json
import math
from pathlib import Path
import struct

from .assurance import policy_digest
from .discrete_fields import review as categorical_review
from .evidence import ONTOLOGY, disposition
from .interpreter import CONTRACT_DIGEST
from .native_review import prepare_input
from .review_fusion import _digest, _number, feature_vector

DIMENSION = 2 + 7 * sum(map(len, ONTOLOGY.values()))
FEATURE_DIGEST = hashlib.sha256(b'native59-interpreter57-sameclass19-float32-trees-v1').hexdigest()


def validate_trees(model):
    if (not isinstance(model, dict) or set(model) != {'format','dimension','intercept','learning_rate','trees','sklearn_version'}
            or model['format'] != 'roguard-boosted-score-v1' or type(model['dimension']) is not int
            or model['dimension'] != DIMENSION or not isinstance(model['sklearn_version'], str)
            or not _number(model['intercept']) or abs(model['intercept']) > 100
            or not _number(model['learning_rate']) or not 0 < model['learning_rate'] <= 1
            or not isinstance(model['trees'], list) or not 1 <= len(model['trees']) <= 100):
        raise ValueError('Invalid boosted model schema')
    for tree in model['trees']:
        if not isinstance(tree, dict) or set(tree) != {'feature','threshold','left','right','value'}:
            raise ValueError('Invalid tree schema')
        if not all(isinstance(v,list) for v in tree.values()):raise ValueError('Invalid tree arrays')
        count=len(tree['feature'])
        if not 1 <= count <= 15 or any(len(v)!=count for v in tree.values()):raise ValueError('Invalid tree size')
        if any(not _number(v) or abs(v)>1e6 for key in ['threshold','value'] for v in tree[key]):
            raise ValueError('Invalid tree numeric value')
        seen=set();stack=[(0,0)]
        while stack:
            node,depth=stack.pop()
            if node in seen or depth>3:raise ValueError('Cyclic, shared or overdeep tree')
            seen.add(node)
            left,right,feature=(tree[k][node] for k in ['left','right','feature'])
            if any(type(v) is not int for v in [left,right,feature]):raise ValueError('Invalid tree index')
            if left==right==-1:
                if feature!=-2:raise ValueError('Invalid leaf feature')
            elif (not 0<=feature<DIMENSION or not node<left<count or not node<right<count or left==right):
                raise ValueError('Invalid child or feature index')
            else:stack.extend([(left,depth+1),(right,depth+1)])
        if len(seen)!=count:raise ValueError('Unreachable tree nodes')


def tree_probability(model, vector):
    if len(vector)!=DIMENSION:raise ValueError('Feature dimension differs')
    values=[struct.unpack('f',struct.pack('f',float(v)))[0] for v in vector]
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite feature')
    logit=model['intercept']
    for tree in model['trees']:
        node=0
        while tree['left'][node]!=-1:
            node=tree['left'][node] if values[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
        logit+=model['learning_rate']*tree['value'][node]
    return 1/(1+math.exp(-max(-700,min(700,logit))))


class ThreeExpertReview:
    def __init__(self, artifact, *, expected_artifact_sha256):
        with Path(artifact).open('rb') as stream:raw=stream.read(2_000_001)
        if len(raw)>2_000_000 or hashlib.sha256(raw).hexdigest()!=_digest(expected_artifact_sha256):
            raise ValueError('Artifact size or identity differs')
        spec=json.loads(raw)
        required={'format','language','native_artifact_sha256','field_artifact_sha256','interpreter_archive_sha256',
            'interpreter_contract_digest','feature_contract_digest','policy_digest','selection_sha256','model','threshold'}
        if not isinstance(spec,dict) or set(spec)!=required or spec['format']!='roguard-three-expert-artifact-v1':
            raise ValueError('Unsupported three-expert artifact')
        if (spec['language'] not in ['ro','uk'] or spec['interpreter_contract_digest']!=CONTRACT_DIGEST
                or spec['feature_contract_digest']!=FEATURE_DIGEST or spec['policy_digest']!=policy_digest()):
            raise ValueError('Model contract differs')
        for key in ['native_artifact_sha256','field_artifact_sha256','interpreter_archive_sha256','selection_sha256']:_digest(spec[key])
        if not _number(spec['threshold']) or not 0<=spec['threshold']<=1:raise ValueError('Invalid threshold')
        validate_trees(spec['model'])
        self._spec=spec;self.identity=expected_artifact_sha256

    def predict(self, payload, native, semantic, interpreter, *, focus=None):
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
        checked = categorical_review(payload, semantic, expected_model_identity=spec['field_artifact_sha256'], focus=focus)
        abstention = native_score is None or not interpreter['valid_label_tokens'] or checked['predicted'] is None
        score = None
        if not abstention:
            vector = feature_vector(native_score, interpreter)
            proposal = int(disposition({k: {'label': v} for k,v in interpreter['interpretation'].items()})[0] != 'CONTINUE')
            if type(interpreter.get('predicted')) is not int or interpreter['predicted'] != proposal:
                raise ValueError('Interpreter policy proposal differs')
            marginal = feature_vector(native_score, dict(interpretation=semantic['interpretation'],
                conditional_field_probabilities=semantic['field_probabilities']))
            width = sum(map(len, ONTOLOGY.values()))
            vector = marginal + vector[2:] + [a*b for a,b in zip(marginal[2:2+width],vector[2:2+width])]
            score = tree_probability(spec['model'], vector)

        predicted = None if score is None else int(score >= spec['threshold'])
        return dict(format='roguard-three-expert-review-v1', language=spec['language'], focus=focus, input_digest=context,
                    model_identity=self.identity, native_model_identity=spec['native_artifact_sha256'], field_model_identity=spec['field_artifact_sha256'],
                    interpreter_model_identity=spec['interpreter_archive_sha256'], feature_contract_digest=FEATURE_DIGEST,
                    predicted=predicted, score=score, decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',
                    reason='COMPONENT_ABSTENTION' if predicted is None else 'FUSION_REVIEW_PROPOSAL',
                    review_requested=predicted != 0, score_is_calibrated=False, reports_authenticated=False,
                    text_in_report=False, establishes_safety=False, permits_external_action=False, permits_notification=False)
