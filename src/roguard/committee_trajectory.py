"""Experimental long-conversation interface for the fixed review committee."""
from copy import deepcopy
import gc

from .assurance import assure, digest, policy_digest
from .committee_bundle import load_committee
from .gemma4_committee import Gemma4ReviewCommittee
from .gemma4_review import _LOCK
from .interpreter import prepare_input
from .trajectory_data import input_identity, prefix, validate_conversation
from .trajectory_model import TrajectoryModel
from .trajectory_neural import factors_from_logits, prepare_memory
from .trajectory_sets import predict_set


class CommitteeTrajectoryModel:
    """Bound source selection, binary committee and optional research field sets.

    This class does not qualify the response checker or a complete v10 release.
    A negative current-prefix result cannot resolve a durable session review.
    """
    @classmethod
    def from_bundle(cls, path, *, expected_sha256, base, selector, budget=1200):
        components = {lang: load_committee(path, expected_sha256=expected_sha256, base=base, language=lang)
                      for lang in ('ro', 'uk')}
        return cls(components, selector, budget=budget)

    def __init__(self, components, selector, *, budget=1200):
        if (not isinstance(components, dict) or set(components) != {'ro', 'uk'}
                or any(not isinstance(model, Gemma4ReviewCommittee) or model.language != lang
                       for lang, model in components.items())):
            raise ValueError('Expected both configured committee languages')
        if not isinstance(selector, TrajectoryModel):
            raise ValueError('Expected a checksum-bound trajectory selector')
        if type(budget) is not int or not 128 <= budget <= 12000:
            raise ValueError('Evidence budget must be 128 to 12000 characters')
        self.components = dict(components)
        # Own the selector configuration so caller mutations cannot silently
        # change memory rendering under a previously recorded identity.
        self.selector = deepcopy(selector); self.budget = budget; self._active = None
        self.identity = digest({'format': 'roguard-committee-trajectory-research-1',
            'components': {lang: model.identity for lang, model in components.items()},
            'selector': selector.identity, 'memory_budget': budget,
            'selection': 'all-prefixes-preserve-complete-turns-v1', 'semantic_provider': 'attention'})

    def pipeline_digest(self):
        return digest({'model': self.identity, 'policy': policy_digest(),
                       'field_distribution': 'independent-six-fields-v1'})

    def predictor_identity(self, *, calibration=None, plan=False):
        if type(plan) is not bool or (plan and calibration is None):
            raise ValueError('Planning requires bound calibration')
        if calibration is not None:
            for lang in ('ro', 'uk'):
                predict_set([1/432]*432, calibration, language=lang,
                            pipeline_digest=self.pipeline_digest(), supported=False)
        return digest({'pipeline': self.pipeline_digest(),
            'calibration': calibration['digest'] if calibration is not None else None, 'plan': plan})

    def predict(self, payload, *, focus=None, calibration=None, plan=False):
        validate_conversation(payload)
        payload = prefix(payload, len(payload['turns'])-1 if focus is None else focus)
        if payload['turns'][-1]['role'] != 'minor':
            raise ValueError('Focal turn must be a minor message')
        binding = self.predictor_identity(calibration=calibration, plan=plan)
        view, inference = prepare_memory(payload, self.selector, self.budget, preserve_complete=True)
        supported = not view['mandatory_overflow'] and bool(inference['turns']) and len(inference['turns']) <= 12
        lang = payload['language']; component = self.components[lang]
        with _LOCK:
            if supported:
                if self._active is not None and self._active != lang:
                    self.components[self._active]._engine = None
                    gc.collect()
                self._active = lang
                raw = component.predict(inference)
                if raw['input_digest'] != prepare_input(inference)[2] or raw['model_identity'] != component.identity:
                    raise ValueError('Committee output belongs to another prefix or model')
            else:
                raw = {'predicted': None, 'score': None, 'decision': 'clarify',
                    'reason': 'EVIDENCE_VIEW_EXCEEDS_COMMITTEE_CONTRACT', 'review_requested': True,
                    'components': None, 'component_identities': dict(component.component_identities),
                    'component_disagreement': None, 'score_kind': 'binary_unanimity_indicator', 'score_is_calibrated': False}
            probabilities = (factors_from_logits(raw['components']['attention']['field_logits'])
                             if raw['components'] is not None else [1/432]*432)
            result = {**raw, 'format': 'roguard-committee-trajectory-research-1',
                'language': lang, 'focus': len(payload['turns'])-1, 'input_digest': input_identity(payload),
                'model_identity': self.identity, 'committee_identity': component.identity,
                'pipeline_digest': self.pipeline_digest(), 'predictor_identity': binding,
                'inference_input_digest': input_identity(inference) if supported else None,
                'used_memory': True, 'probabilities': probabilities,
                'evidence': [{k: span[k] for k in ('turn', 'start', 'end')} for span in view['spans']],
                'evidence_semantics_verified': False, 'qualified': False, 'text_in_report': False,
                'establishes_safety': False, 'permits_external_action': False, 'permits_notification': False}
            if calibration is not None:
                sets = predict_set(probabilities, calibration, language=lang,
                                   pipeline_digest=self.pipeline_digest(), supported=raw['predicted'] is not None)
                result['interpretation_set'] = sets
                if plan:
                    result['assurance'] = assure({'language': lang, 'context_digest': result['input_digest'],
                        'hypotheses': sets['hypotheses'], 'model_digests': [self.identity],
                        'review_floor': raw['predicted'] != 0})
            return result


def main():
    import argparse
    import json
    from pathlib import Path
    import sys
    from .native_review import _read
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--selector', type=Path, required=True)
    parser.add_argument('--selector-sha256', required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--focus', type=int)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    try:
        raw = _read(args.input, 2_500_000) if args.input else sys.stdin.buffer.read(2_500_001)
        if len(raw) > 2_500_000:
            raise ValueError('Conversation JSON is too large')
        payload = json.loads(raw); validate_conversation(payload)
        calibration = json.loads(_read(args.calibration, 65_536)) if args.calibration else None
        selector = TrajectoryModel(args.selector, expected_sha256=args.selector_sha256)
        model = CommitteeTrajectoryModel.from_bundle(args.bundle, expected_sha256=args.sha256, base=args.base, selector=selector)
        print(json.dumps(model.predict(payload, focus=args.focus, calibration=calibration, plan=args.plan), ensure_ascii=True))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
