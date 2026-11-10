"""Opt-in, hash-bound v10 review of frozen Gemma4 features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ._portable_rbf_v10 import PortableRBF
from .gemma4_review import Gemma4ReviewModel, _LOCK, _score
from .gemma4_review_v10 import V10FeatureEngine
from .interpreter import prepare_input

PACKAGED_MANIFEST_SHA256 = '52108c04b0313e246c4a63d029e07b8475600acd6fb3241d3d784387b3bb754b'


def _binary_decision(parent_predicted: int, binary_margin: float, concern_margin: float,
                     concern_threshold: float = 0.0) -> int:
    # A strong negative binary margin can clear an out-of-scope report even when
    # the text still mentions a general safety concern. Weaker margins require
    # the concern head to agree, preserving difficult positive parent decisions.
    if parent_predicted == 1 and binary_margin <= 0.05:
        if concern_margin < concern_threshold or binary_margin <= -0.8:
            return 0
    if parent_predicted == 0 and binary_margin >= 0.5 and concern_margin > concern_threshold:
        return 1
    return parent_predicted


class Gemma4RBFV10Model:
    """Run two portable heads on the parent's exact final token feature."""

    def __init__(self, parent: Gemma4ReviewModel, binary_asset, concern_asset, *,
                 binary_sha256: str, concern_sha256: str):
        if not isinstance(parent, Gemma4ReviewModel):
            raise TypeError('Expected a pinned Gemma4 parent')
        self.parent = parent
        self.language = parent.language
        self.binary = PortableRBF(binary_asset, expected_sha256=binary_sha256,
                                  feature_mode='joint')
        self.concern = PortableRBF(concern_asset, expected_sha256=concern_sha256,
                                   feature_mode='joint')
        binding = json.dumps({'format': 'roguard-gemma4-rbf-v10', 'parent': parent.identity,
                              'language': self.language, 'binary': self.binary.identity,
                              'concern': self.concern.identity, 'feature_mode': 'joint',
                              'decision_rule': 'guard-v3-ro0-uk0.05-meal-cue'},
                             sort_keys=True, separators=(',', ':'))
        self.identity = hashlib.sha256(binding.encode('utf-8')).hexdigest()

    @classmethod
    def from_bundle(cls, path, *, expected_sha256):
        from .field_pipeline import _local_path
        from .language_bundle import _read_manifest
        from .native_review import _digest

        path = Path(path).resolve()
        manifest = _read_manifest(path, expected_sha256)
        if (not isinstance(manifest, dict) or set(manifest) != {'format', 'parent_bundle_sha256',
                'binary_sha256', 'concern_sha256', 'assets'}
                or manifest['format'] != 'roguard-gemma4-rbf-v10-bundle-v1'
                or not isinstance(manifest['assets'], dict)
                or set(manifest['assets']) != {'parent_bundle', 'binary', 'concern'}):
            raise ValueError('Unsupported Gemma4 RBF bundle')
        assets = {name: _local_path(path.parent, location) for name, location in manifest['assets'].items()}
        parent = Gemma4ReviewModel.from_bundle(assets['parent_bundle'],
                                               expected_sha256=_digest(manifest['parent_bundle_sha256']))
        return cls(parent, assets['binary'], assets['concern'],
                   binary_sha256=_digest(manifest['binary_sha256']),
                   concern_sha256=_digest(manifest['concern_sha256']))

    @classmethod
    def from_v9_bundle(cls, bundle, *, language):
        """Use the installed, pinned v9 backbone with packaged v10 heads."""
        from .language_bundle import _read_manifest

        if language not in ('ro', 'uk'):
            raise ValueError('Expected Romanian or Ukrainian language')
        assets = Path(__file__).with_name('v10_models')
        manifest = _read_manifest(assets / 'manifest.json', PACKAGED_MANIFEST_SHA256)
        if (not isinstance(manifest, dict) or set(manifest) != {
                'format', 'feature_mode', 'parent_profile', 'parent_root_bundle_sha256',
                'parent_language_bundle_sha256', 'models', 'engineering_fit_report_sha256',
                'engineering_policy_recount_sha256', 'independent_human_validation'}
                or manifest['format'] != 'roguard-gemma4-rbf-v10-packaged-assets-1'
                or manifest['feature_mode'] != 'joint'
                or manifest['parent_profile'] != 'v9-gemma4-20261019'
                or set(manifest['parent_language_bundle_sha256']) != {'ro', 'uk'}
                or set(manifest['models']) != {'ro', 'uk'}
                or any(set(manifest['models'][lang]) != {'binary', 'concern'}
                       for lang in ('ro', 'uk'))
                or manifest['independent_human_validation'] is not False):
            raise ValueError('Unsupported packaged v10 model contract')
        bundle = Path(bundle).resolve()
        _read_manifest(bundle, manifest['parent_root_bundle_sha256'])
        parent_path = bundle.parent / (language + '-bundle.json')
        parent = Gemma4ReviewModel.from_bundle(
            parent_path,
            expected_sha256=manifest['parent_language_bundle_sha256'][language])
        return cls(parent, assets / (language + '-binary.npz'),
                   assets / (language + '-concern.npz'),
                   binary_sha256=manifest['models'][language]['binary'],
                   concern_sha256=manifest['models'][language]['concern'])

    @classmethod
    def from_installed_v9(cls, *, language, cache=None):
        """Load from a prior explicit `roguard-fetch fetch` installation."""
        from .model_install import cache_directory, install, profile

        raw, digest = profile('v9')
        bundle, _ = install(raw, digest, cache_directory() if cache is None else cache,
                            download=False)
        return cls.from_v9_bundle(bundle, language=language)

    def predict(self, payload, *, focus=None):
        return self.predict_many([payload], focuses=[focus])[0]

    def interpret(self, payload, *, focus=None):
        return self.interpret_many([payload], focuses=[focus])[0]

    def predict_many(self, payloads, *, focuses=None):
        return self.interpret_many(payloads, focuses=focuses)

    def interpret_many(self, payloads, *, focuses=None):
        from .evidence import ONTOLOGY
        from .review_scope_v10 import correct_logits, explicit_meal_deprivation

        if not isinstance(payloads, list) or not 1 <= len(payloads) <= 128:
            raise ValueError('Expected one to 128 conversations')
        if focuses is None:
            focuses = [None] * len(payloads)
        if not isinstance(focuses, list) or len(focuses) != len(payloads):
            raise ValueError('Focal indices do not match the batch')
        prepared = [prepare_input(payload, focus) for payload, focus in zip(payloads, focuses)]
        if any(prefix['language'] != self.language for prefix, _, _ in prepared):
            raise ValueError('Gemma4 adapter language differs')
        with _LOCK:
            if self.parent._engine is None:
                self.parent._engine = V10FeatureEngine(*self.parent._args)
            outputs = self.parent._engine.scores([p[0] for p in prepared], self.parent._spec,
                                                 return_logits=True, return_features=True,
                                                 return_focal=True)
        if not isinstance(outputs, list) or len(outputs) != len(prepared):
            raise ValueError('Incomplete Gemma4 RBF review output')
        results = []
        for item, (prefix, focus, digest) in zip(outputs, prepared):
            shared = dict(format='roguard-gemma4-rbf-v10', language=self.language, focus=focus,
                          input_digest=digest, model_identity=self.identity, score=None,
                          score_is_calibrated=False, text_in_report=False, establishes_safety=False,
                          permits_external_action=False, permits_notification=False)
            if item is None:
                results.append(dict(shared, predicted=None, decision='clarify',
                                    reason='INPUT_EXCEEDS_MODEL_CONTEXT', fields=None,
                                    review_requested=True, scope_evidence=None))
                continue
            import numpy as np
            raw, final, focal = item
            feature = np.concatenate((final, focal)).astype(np.float32, copy=False)
            _, evidence = correct_logits(raw, text=prefix['turns'][-1]['text'],
                                         language=self.language, spec=self.parent._spec, score_fn=_score)
            parent_predicted = int(evidence['score'] >= self.parent._spec['threshold'])
            binary_margin = self.binary.margin(feature)
            concern_margin = self.concern.margin(feature)
            concern_threshold = 0.05 if self.language == 'uk' else 0.0
            predicted = _binary_decision(parent_predicted, binary_margin, concern_margin,
                                         concern_threshold)
            meal_deprivation = explicit_meal_deprivation(prefix['turns'][-1]['text'], self.language)
            if meal_deprivation:
                predicted = 1
            field_ids = list(evidence['fields'])
            field_ids[5] = int(concern_margin >= concern_threshold)
            if meal_deprivation:
                field_ids[5] = 1
            fields = {field: ONTOLOGY[field][index] for field, index in zip(ONTOLOGY, field_ids)}
            results.append(dict(shared, predicted=predicted,
                                decision='review' if predicted else 'no_signal',
                                reason='MODEL_REVIEW_PROPOSAL' if predicted else 'NO_SIGNAL_DETECTED',
                                fields=fields, review_requested=bool(predicted),
                                binary_margin=binary_margin, concern_margin=concern_margin,
                                parent_predicted=parent_predicted,
                                scope_evidence={'cues': evidence['cues'],
                                                'binary_guard_rejected': evidence['binary_guard_rejected'],
                                                'explicit_meal_deprivation': meal_deprivation}))
        return results
