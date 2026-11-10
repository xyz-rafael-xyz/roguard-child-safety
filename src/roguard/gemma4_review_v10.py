"""Opt-in feature capture and structured interpretation over the pinned v9 runtime."""
from __future__ import annotations

import numpy as np

from .gemma4_contract import PREFIX_TOKENS, classifier_tokens
from .gemma4_review import Gemma4ReviewModel, _Engine, _LOCK, _score
from .interpreter import prepare_input


class V10FeatureEngine(_Engine):
    """Reuse the v9 loader and extract additional features without editing v9."""

    def scores(self, prefixes, spec, *, return_logits=False, return_features=False,
               return_focal=False):
        from .gemma4_attention import capture, capture_with_prefix, replay
        if (return_features and not return_logits) or (return_focal and not return_features):
            raise ValueError('Focal feature capture requires raw logits and final features')
        if return_focal:
            from .focal_reader_residual import native_focal_mean
            from .role_alignment import align
        mx = self.mx
        sequences = [classifier_tokens(self.tokenizer, p) for p in prefixes]
        common = len(PREFIX_TOKENS)
        cache = None
        if any(common < len(s) <= 1024 and s[:common] == PREFIX_TOKENS for s in sequences):
            cache = self.make_cache(self.model.base)
            mx.eval(self.model.base.model(mx.array([PREFIX_TOKENS]), cache=cache))
        output = []
        for prefix, sequence in zip(prefixes, sequences):
            if len(sequence) > 1024:
                output.append(None)
                continue
            if cache is not None and len(sequence) > common and sequence[:common] == PREFIX_TOKENS:
                state = capture_with_prefix(self.model.base, mx.array([sequence[common:]]), cache, start=24)
            else:
                state = capture(self.model.base, mx.array([sequence]), start=24)
            features = replay(self.model.base, state)[:, 0, :]
            logits = self.model.heads[spec['language']](features)
            mx.eval(logits)
            if return_features:
                values = (np.asarray(logits[0], dtype=np.float32).copy(),
                          np.asarray(features[0], dtype=np.float32).copy())
                if return_focal:
                    alignment = align(prefix, getattr(self.tokenizer, '_tokenizer', self.tokenizer))
                    if alignment['classifier_token_ids'] != sequence:
                        raise ValueError('Focal token alignment differs')
                    converted = state['shared'][23][1].astype(mx.float32)
                    mx.eval(converted)
                    focal, present = native_focal_mean(mx, converted, alignment)
                    if not present:
                        raise ValueError('Focal minor content absent')
                    values += (focal,)
                output.append(values)
            else:
                output.append(np.asarray(logits[0], dtype=np.float32).copy() if return_logits else _score(logits[0], spec))
            del state, features, logits
            mx.clear_cache()
        return output


class Gemma4StructuredV10Model(Gemma4ReviewModel):
    """Opt-in structured readout; the inherited v9 prediction remains available."""

    def interpret(self, payload, *, focus=None):
        return self.interpret_many([payload], focuses=[focus])[0]

    def interpret_many(self, payloads, *, focuses=None):
        from .evidence import ONTOLOGY
        from .review_scope_v10 import correct_logits

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
            if self._engine is None:
                self._engine = V10FeatureEngine(*self._args)
            logits = self._engine.scores([prefix for prefix, _, _ in prepared], self._spec,
                                         return_logits=True)
        if not isinstance(logits, list) or len(logits) != len(prepared):
            raise ValueError('Incomplete structured review output')
        results = []
        for raw, (prefix, focus, identity) in zip(logits, prepared):
            if raw is None:
                results.append(dict(format='roguard-structured-review-v10', language=self.language, focus=focus,
                                    input_digest=identity, model_identity=self.identity, score=None, predicted=None,
                                    decision='clarify', reason='INPUT_EXCEEDS_MODEL_CONTEXT', fields=None,
                                    scope_evidence=None, review_requested=True, score_is_calibrated=False,
                                    text_in_report=False, establishes_safety=False, permits_external_action=False,
                                    permits_notification=False))
                continue
            _, evidence = correct_logits(raw, text=prefix['turns'][-1]['text'], language=self.language,
                                         spec=self._spec, score_fn=_score)
            score = evidence['score']
            predicted = int(score >= self._spec['threshold'])
            fields = {field: ONTOLOGY[field][index] for field, index in zip(ONTOLOGY, evidence['fields'])}
            results.append(dict(format='roguard-structured-review-v10', language=self.language, focus=focus,
                                input_digest=identity, model_identity=self.identity, score=score, predicted=predicted,
                                decision='review' if predicted else 'no_signal',
                                reason='MODEL_REVIEW_PROPOSAL' if predicted else 'NO_SIGNAL_DETECTED', fields=fields,
                                scope_evidence={'cues': evidence['cues'],
                                                'binary_guard_rejected': evidence['binary_guard_rejected']},
                                review_requested=predicted != 0, score_is_calibrated=False,
                                text_in_report=False, establishes_safety=False,
                                permits_external_action=False, permits_notification=False))
        return results
