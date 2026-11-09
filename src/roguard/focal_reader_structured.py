"""Focal-reader fields with source evidence and pipeline-bound interpretation sets.

This adapter grants no calibration, release qualification, or external actions.
The frozen focal reader supplies the same-call logits; memory selection and
interpretation-set construction retain the existing structured API semantics.
"""
from copy import deepcopy
import gzip
import json
import math
import struct

from roguard.assurance import digest, policy_digest
from roguard.evidence import ONTOLOGY, _complete_action
from roguard.experimental_structured import (check_evidence, rendering_segments,
    SELECTOR_SHA, MASS_ATOL, canonical, sha, require, InactiveLanguageError)
from roguard.focal_reader_residual import (FocalReaderResidualModel,
    V9 as WEIGHTS9, V9_SPEC as SPEC9, validate_manifest)
from roguard.gemma4_review import _score, _LOCK
from roguard.trajectory_data import validate_conversation, prefix, input_identity
from roguard.trajectory_memory import units, verify_view, MEMORY_CONTRACT
from roguard.trajectory_neural import prepare_memory, factors_from_logits
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_sets import WORLDS, predict_set
from roguard.v10_closure import FocalReaderClosure

BATCH_ROWS = 96
ROUTING = dict(routes=['focal_reader', 'focal_reader_unsupported'],
    maximum_classifier_tokens=1024, fallback='none')


class FocalReaderStructuredWrapper:
    """Use the explicit focal component for every supported retained view."""
    def __init__(self, components, selector, closure, *, active_language,
                 mode='direct-first', budget=1200):
        require(active_language in ('ro', 'uk') and type(components) is dict and
                set(components) == {active_language}, 'One explicit active language required')
        require(type(mode) is str and mode in ('direct-first', 'uniform') and
                type(budget) is int and budget == 1200, 'Frozen memory configuration')
        require(type(closure) is FocalReaderClosure, 'Verified focal-reader closure required')
        closure.verify()
        require(type(selector) is TrajectoryModel and
                selector.identity == closure.asset_sha('selector') == SELECTOR_SHA,
                'Frozen trajectory selector required')
        require(closure.asset_path('selector').stat().st_size <= 20_000_000, 'Selector exceeds bound')
        with gzip.open(closure.asset_path('selector'), 'rb') as stream:
            raw = stream.read(40_000_001)
        require(len(raw) <= 40_000_000 and canonical(json.loads(raw)) == canonical(selector.spec),
                'Loaded selector specification differs')
        self.components = dict(components)
        self.selector = deepcopy(selector)
        self.closure = closure
        self.active_language, self.mode, self.budget = active_language, mode, budget
        self.specs, self.manifests, self._manifest_values = {}, {}, {}
        for language in ('ro', 'uk'):
            require(closure.asset_sha(language + '_spec9') == SPEC9[language] and
                    closure.asset_sha(language + '_adapter9') == WEIGHTS9[language],
                    'Original-v9 asset identity')
            spec = json.loads(closure.asset_path(language + '_spec9').read_bytes())
            require(spec['language'] == language and spec['policy_digest'] == policy_digest() and
                    spec['mode'] == ('policy' if language == 'ro' else 'binary'),
                    'Own-v9 score specification')
            manifest = validate_manifest(json.loads(closure.asset_path(language + '_manifest').read_bytes()))
            require(manifest['language'] == language and
                    all(manifest['arrays'][role] == closure.asset_sha(language + '_' + role)
                        for role in ('reader_weights', 'reader_stats', 'residual_weights', 'residual_stats')),
                    'Focal manifest and closure assets differ')
            self.specs[language] = spec
            self.manifests[language] = closure.asset_sha(language + '_manifest')
            self._manifest_values[language] = manifest
        self._spec_state = digest(self.specs)
        self._manifest_state = digest(self._manifest_values)
        self._selector_state = digest(self.selector.spec)
        self._component = self.components[active_language]
        self.contract = dict(format='roguard-focal-reader-structured-pipeline-1', closure=closure.receipt,
            wrapper_source_sha256=sha(__file__), ontology={k: list(v) for k, v in ONTOLOGY.items()},
            policy_digest=policy_digest(), mode=mode, budget=budget, memory_contract=MEMORY_CONTRACT,
            preserve_complete=True, selector_sha256=self.selector.identity, routing=deepcopy(ROUTING),
            maximum_direct_turns=12, public_batch_rows=BATCH_ROWS,
            distribution_contracts={'ro': 'independent-six-fields-v1;policy-score',
                'uk': 'independent-six-fields-v1;separate-legacy-binary-score',
                'unsupported': 'full432-no-confidence'})
        self.identity = digest(self.contract)
        self._active = False
        self._guard()

    def _guard(self, *, full=False):
        require(set(self.components) == {self.active_language} and
                self.components[self.active_language] is self._component, 'Active component changed')
        component = self._component
        require(type(component) is FocalReaderResidualModel and component.language == self.active_language and
                getattr(component.predict_details, '__func__', None) is FocalReaderResidualModel.predict_details,
                'Reviewed focal-reader component required')
        require(component.identity == self.manifests[self.active_language] ==
                self.closure.asset_sha(self.active_language + '_manifest') and
                canonical(component.manifest) == canonical(self._manifest_values[self.active_language]),
                'Focal component manifest differs')
        require(component._parent.identity == self.closure.asset_sha(self.active_language + '_spec9') and
                canonical(component._parent._spec) == canonical(self.specs[self.active_language]),
                'Focal parent score specification differs')
        require(type(self.selector) is TrajectoryModel and self.selector.identity == SELECTOR_SHA and
                getattr(self.selector.relevance, '__func__', None) is TrajectoryModel.relevance,
                'Selector implementation changed')
        require(self.mode == self.contract['mode'] and self.budget == self.contract['budget'] and
                BATCH_ROWS == self.contract['public_batch_rows'] and ROUTING == self.contract['routing'],
                'Memory/routing/batch configuration changed')
        require(digest(self.contract) == self.identity and sha(__file__) == self.contract['wrapper_source_sha256'] and
                digest(self.specs) == self._spec_state and digest(self.selector.spec) == self._selector_state and
                digest(self._manifest_values) == self._manifest_state, 'Structured pipeline state changed')
        (self.closure.verify if full else self.closure.cheap_verify)()

    @staticmethod
    def _f32_tuple(value, width):
        if type(value) is not tuple or len(value) != width:
            return False
        try:
            return all(type(v) is float and math.isfinite(v) and
                       struct.unpack('<f', struct.pack('<f', v))[0].hex() == v.hex() for v in value)
        except (OverflowError, struct.error):
            return False

    def _detail(self, inference, detail):
        require(getattr(detail, '_fields', None) ==
                ('response', 'logits', 'reader_features', 'focal_mean', 'focal_present'),
                'Exact focal detailed prediction required')
        language = inference['language']; spec = self.specs[language]
        raw = deepcopy(detail.response); z = detail.logits; unsupported = z is None
        route = 'focal_reader_unsupported' if unsupported else 'focal_reader'
        if unsupported:
            require(detail.reader_features is None and detail.focal_mean is None and
                    detail.focal_present is None, 'Unsupported input cannot contain features')
            score = predicted = probabilities = mass = None
        else:
            require(self._f32_tuple(z, 19) and self._f32_tuple(detail.reader_features, 6) and
                    self._f32_tuple(detail.focal_mean, 1024) and type(detail.focal_present) is bool,
                    'Immutable finite F32 logits and focal-reader features required')
            require(detail.reader_features[5] in (0., 1.) and
                    (detail.reader_features[5] == 1. or not any(detail.reader_features)),
                    'Absent reader must be neutral')
            require(detail.focal_present or not any(detail.focal_mean), 'Absent focal must be neutral')
            score = float(_score(z, spec)); predicted = int(score >= spec['threshold'])
            probabilities = factors_from_logits(z)
            semantic = math.fsum(v for v, world in zip(probabilities, WORLDS)
                                 if _complete_action(world) != 'CONTINUE')
            equal = language == 'ro'
            if equal:
                require(abs(semantic - score) <= MASS_ATOL, 'RO semantic policy mass differs')
            mass = dict(semantic_review_mass=semantic, score_mass_equivalence_required=equal,
                        semantic_vs_binary_difference=abs(semantic - score))
        expected = dict(format='roguard-canonical-focal-reader-review-v1', language=language,
            focus=len(inference['turns']) - 1, input_digest=input_identity(inference),
            model_identity=self.manifests[language], score=score, predicted=predicted,
            decision='clarify' if unsupported else 'review' if predicted else 'no_signal',
            reason='INPUT_EXCEEDS_MODEL_CONTEXT' if unsupported else 'MODEL_REVIEW_PROPOSAL' if predicted else 'NO_SIGNAL_DETECTED',
            review_requested=predicted != 0, score_is_calibrated=False, text_in_report=False,
            establishes_safety=False, permits_external_action=False, permits_notification=False)
        require(canonical(raw) == canonical(expected), 'Full same-call focal response differs')
        return raw, probabilities, route, mass, z

    def _native_batches(self, prepared, indices):
        for lo in range(0, len(indices), BATCH_ROWS):
            self._guard()
            ii = indices[lo:lo + BATCH_ROWS]
            payloads = [prepared[i]['inference'] for i in ii]
            details = self._component.predict_details(payloads,
                focuses=[len(p['turns']) - 1 for p in payloads])
            require(type(details) is list and len(details) == len(ii), 'Complete native batch required')
            for i, detail in zip(ii, details):
                prepared[i]['detail'] = self._detail(prepared[i]['inference'], detail)
            self._guard()

    def predict(self, payload, *, focus=None, calibration=None):
        return self.predict_many([dict(payload=payload, focus=focus, calibration=calibration)])[0]

    def predict_many(self, requests):
        require(type(requests) is list and requests, 'Nonempty request list required')
        prepared = []
        for request in requests:
            require(type(request) is dict and 'payload' in request and
                    set(request) <= {'payload', 'focus', 'calibration'}, 'Exact structured request')
            payload = request['payload']; validate_conversation(payload)
            if payload['language'] != self.active_language:
                raise InactiveLanguageError('Language is not loaded in this worker: ' + payload['language'])
            focus = request.get('focus')
            focus = len(payload['turns']) - 1 if focus is None else focus
            original = deepcopy(prefix(payload, focus))
            require(original['turns'][-1]['role'] == 'minor', 'Focal minor required')
            prepared.append(dict(original=original, focus=focus, inference=original, detail=None,
                direct=None, view=None, segments=[], unsupported_reason=None,
                calibration=deepcopy(request.get('calibration'))))
        results = []
        with _LOCK:
            require(not self._active, 'Structured batch already active')
            self._guard(full=True); self._active = True; failure = None
            try:
                direct_indices = [i for i, p in enumerate(prepared)
                                  if self.mode == 'direct-first' and len(p['original']['turns']) <= 12]
                self._native_batches(prepared, direct_indices)
                memory_indices = []; direct_set = set(direct_indices)
                for i, p in enumerate(prepared):
                    if i in direct_set:
                        p['direct'] = p['detail']
                    if self.mode == 'uniform' or len(p['original']['turns']) > 12 or (
                            p['direct'] is not None and p['direct'][2] == 'focal_reader_unsupported'):
                        view, inference = prepare_memory(p['original'], self.selector, self.budget, preserve_complete=True)
                        verify_view(p['original'], view); p.update(view=view, inference=inference)
                        if view['mandatory_overflow'] or not inference['turns'] or len(inference['turns']) > 12:
                            p['unsupported_reason'] = 'MANDATORY_MEMORY_OVERFLOW' if view['mandatory_overflow'] else 'MEMORY_VIEW_EXCEEDS_ORIGINAL_TURN_CONTRACT'
                            p['detail'] = None
                        else:
                            p['segments'] = rendering_segments(p['original'], view, inference)
                            memory_indices.append(i)
                self._native_batches(prepared, memory_indices)
                for p in prepared:
                    results.append(self._materialize(p))
                self._guard()
            except BaseException as error:
                failure = error
                raise
            finally:
                try:
                    self._guard(full=True)
                except BaseException as closing:
                    if failure is not None:
                        raise BaseExceptionGroup('Structured inference and closing failed', [failure, closing])
                    raise
                finally:
                    self._active = False
            for result in results:
                result['closure_verification'] = dict(state='PASS',
                    scope='full_disk_closure_at_batch_boundaries;focal_reader_batches_at_most96')
            return results

    def _materialize(self, p):
        original, inference, view = p['original'], p['inference'], p['view']
        raw, probabilities, route, mass, logits = p['detail'] if p['detail'] is not None else (None, None, 'memory_unsupported', None, None)
        evidence = [{k: s[k] for k in ('turn', 'start', 'end')} for s in (view['spans'] if view is not None else units(original))]
        check_evidence(original, evidence)
        score, predicted = (None, None) if raw is None else (raw['score'], raw['predicted'])
        supported = predicted is not None
        sets = None
        if not supported:
            probabilities = [1 / 432] * 432
            sets = dict(hypotheses=deepcopy(WORLDS), reason='UNSUPPORTED_OR_EMPTY_SET', calibration_digest=None,
                        coverage_assumptions_verified=False, outside_ontology_ruled_out=False)
        if p['calibration'] is not None:
            sets = predict_set(probabilities, p['calibration'], language=original['language'],
                               pipeline_digest=self.identity, supported=supported)
        result = dict(format='roguard-full-route-structured-prediction-research-1', language=original['language'],
            focus=p['focus'], input_digest=input_identity(original),
            inference_input_digest=None if not inference['turns'] else input_identity(inference),
            model_identity=self.identity, pipeline_digest=self.identity, predicted=predicted, score=score,
            native_response=raw, field_logits=logits, direct_unsupported_response=None if p['direct'] is None or
            p['direct'][2] != 'focal_reader_unsupported' else p['direct'][0], route=route,
            semantic_score_relation=mass, probabilities=probabilities,
            probabilities_source='SAME_FORWARD_ROUTE_LOGITS' if supported else 'UNSUPPORTED_UNIFORM_PLACEHOLDER',
            evidence=evidence, evidence_scope='retained_source_units_not_verified_semantic_support',
            used_memory=view is not None, memory=view, inference_to_original_segments=p['segments'],
            unsupported_reason=p['unsupported_reason'], qualified=False, establishes_safety=False,
            permits_external_action=False)
        if sets is not None:
            result['interpretation_set'] = sets
        return result
