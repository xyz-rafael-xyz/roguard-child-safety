"""Experimental field-wise fusion of two explicit local interpretation reports."""
import hashlib
import json
import math
from pathlib import Path

from .assurance import policy_digest
from .native_fields import FIELDS, factorized_review_mass, pair_probabilities, _check_couplings
from .native_review import prepare_input


class FieldReviewFusion:
    """Combine bound neural field reports; hashes do not authenticate producers."""

    def __init__(self, artifact, *, expected_sha256):
        with Path(artifact).open('rb') as stream:
            raw = stream.read(65_537)
        if len(raw) > 65_536 or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('Field-fusion artifact identity or size differs')
        spec = json.loads(raw)
        keys = {'format', 'language', 'policy_digest', 'native_heads_sha256',
                'pretrained_heads_sha256', 'native_weights', 'couplings', 'threshold'}
        if (not isinstance(spec, dict) or set(spec) != keys
                or spec['format'] != 'roguard-field-fusion-artifact-v1'
                or spec['language'] not in ('ro', 'uk') or spec['policy_digest'] != policy_digest()
                or not isinstance(spec['native_weights'], dict) or set(spec['native_weights']) != set(FIELDS)):
            raise ValueError('Unsupported field-fusion contract')
        for key in ('native_heads_sha256', 'pretrained_heads_sha256'):
            value = spec[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid source model identity')
        for value in [spec['threshold'], *spec['native_weights'].values()]:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid field-fusion parameter')
        _check_couplings(spec['couplings'])
        self._spec = spec
        self.identity = expected_sha256

    def predict(self, payload, native_report, pretrained_report, *, focus=None):
        _, focus, identity = prepare_input(payload, focus)
        if payload['language'] != self._spec['language']:
            raise ValueError('Conversation language differs from fusion model')
        values = []
        for report, key in ((native_report, 'native_heads_sha256'),
                            (pretrained_report, 'pretrained_heads_sha256')):
            if (not isinstance(report, dict) or report.get('format') != 'roguard-native-fields-v1'
                    or report.get('model_identity') != self._spec[key]
                    or report.get('input_digest') != identity or report.get('focus') != focus
                    or report.get('language') != self._spec['language']
                    or report.get('policy_digest') != self._spec['policy_digest']):
                raise ValueError('Neural field report binding differs')
            probabilities = report.get('field_probabilities')
            if probabilities is None:
                if report.get('score') is not None or report.get('predicted') is not None:
                    raise ValueError('Inconsistent source abstention')
                values.append(None)
            else:
                factorized_review_mass(probabilities)  # Validate all six vectors.
                score = report.get('score')
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Invalid source score')
                values.append(probabilities)
        mixed = None
        if all(v is not None for v in values):
            mixed = {field: [self._spec['native_weights'][field]*x +
                            (1-self._spec['native_weights'][field])*y
                            for x, y in zip(values[0][field], values[1][field])] for field in FIELDS}
        score = None if mixed is None else factorized_review_mass(mixed, couplings=self._spec['couplings'])
        predicted = None if score is None else int(score >= self._spec['threshold'])
        return dict(format='roguard-field-fusion-v1', language=self._spec['language'], focus=focus,
            input_digest=identity, model_identity=self.identity, policy_digest=self._spec['policy_digest'],
            source_model_identities=[self._spec['native_heads_sha256'], self._spec['pretrained_heads_sha256']],
            field_probabilities=mixed, pair_probabilities=None if mixed is None else pair_probabilities(mixed, self._spec['couplings']),
            score=score, predicted=predicted,
            decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',
            review_requested=predicted != 0, probabilities_are_calibrated=False,
            field_independence_assumed=self._spec['couplings'] is None, factor_independence_assumed=True,
            source_authentication_required=True, establishes_safety=False,
            permits_notification=False, permits_external_action=False, text_in_report=False)
