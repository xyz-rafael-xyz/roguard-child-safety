"""Apply the declared review policy to six explicit argmax interpretations.

This experimental decision path uses the categorical interpretation itself,
not a threshold on integrated probability mass. It grants no external powers.
"""
import hashlib
import json

from .assurance import policy_digest
from .evidence import ONTOLOGY, disposition
from .native_fields import factorized_review_mass
from .native_review import prepare_input

CONTRACT = 'six-head-argmax-first-index-ties-fixed-policy-v1'
CONTRACT_DIGEST = hashlib.sha256(CONTRACT.encode()).hexdigest()


def review(payload, report, *, expected_model_identity, focus=None):
    _, focus, identity = prepare_input(payload, focus)
    if (not isinstance(expected_model_identity, str) or len(expected_model_identity) != 64
            or any(c not in '0123456789abcdef' for c in expected_model_identity)):
        raise ValueError('Expected a trusted model digest')
    if (not isinstance(report, dict) or report.get('format') != 'roguard-native-fields-v1'
            or report.get('model_identity') != expected_model_identity
            or report.get('input_digest') != identity or type(report.get('focus')) is not int
            or report['focus'] != focus or report.get('language') != payload['language']
            or report.get('policy_digest') != policy_digest()
            or report.get('interpretation_is_head_argmax') is not True):
        raise ValueError('Semantic report binding differs')
    for flag in ('probabilities_are_calibrated', 'text_in_report', 'establishes_safety',
                 'permits_external_action', 'permits_notification'):
        if report.get(flag) is not False:
            raise ValueError('Unsupported source claim')
    probabilities = report.get('field_probabilities')
    values = None
    if probabilities is None:
        if any(report.get(key) is not None for key in ('interpretation', 'score', 'predicted')):
            raise ValueError('Inconsistent source abstention')
    else:
        factorized_review_mass(probabilities)  # Validate six complete finite vectors.
        values = {field: labels[max(range(len(labels)), key=probabilities[field].__getitem__)]
                  for field, labels in ONTOLOGY.items()}
        if values != report.get('interpretation'):
            raise ValueError('Interpretation differs from field probabilities')
    predicted = None if values is None else int(disposition({f: {'label': v} for f, v in values.items()})[0] != 'CONTINUE')
    model_identity = hashlib.sha256(json.dumps([expected_model_identity, CONTRACT_DIGEST,
        policy_digest()], separators=(',', ':')).encode()).hexdigest()
    return dict(format='roguard-discrete-fields-v1', language=payload['language'], focus=focus,
        input_digest=identity, model_identity=model_identity, source_model_identity=expected_model_identity,
        policy_digest=policy_digest(), contract_digest=CONTRACT_DIGEST,
        predicted=predicted, interpretation=values, field_probabilities=probabilities,
        decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',
        reason='SOURCE_ABSTENTION' if predicted is None else 'CATEGORICAL_POLICY_PROPOSAL',
        review_requested=predicted != 0, probabilities_are_calibrated=False,
        source_authentication_required=True, text_in_report=False, establishes_safety=False,
        permits_external_action=False, permits_notification=False)


class DiscreteFieldsModel:
    """Wrap one explicit native semantic model; never select or download it."""
    def __init__(self, model):
        self.model = model

    def predict(self, payload, *, focus=None):
        report = self.model.predict(payload, focus=focus)
        return review(payload, report, expected_model_identity=self.model.identity, focus=focus)

    def predict_many(self, payloads, *, focuses=None):
        reports = self.model.predict_many(payloads, focuses=focuses)
        if len(reports) != len(payloads):
            raise ValueError('Incomplete semantic batch')
        if focuses is None:
            focuses = [None] * len(payloads)
        return [review(payload, report, expected_model_identity=self.model.identity, focus=focus)
                for payload, report, focus in zip(payloads, reports, focuses)]
