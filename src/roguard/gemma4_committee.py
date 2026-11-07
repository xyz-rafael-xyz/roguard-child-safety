"""Experimental local review unanimity with one shared frozen encoder body.

This is a binary review component, not a qualified release or safety guarantee.
Every component evaluates the same prefix. No downloads or external actions occur.
"""
from copy import deepcopy
import math

from .assurance import digest
from .gemma4_attention import capture, capture_with_prefix, replay
from .gemma4_contract import PREFIX_TOKENS, classifier_tokens
from .gemma4_review import Gemma4ReviewModel, _Engine, _LOCK, _load_weights, _score
from .interpreter import prepare_input
from .native_review import _read, _sha


COMPONENTS = ('v9', 'readout', 'attention')
ALGORITHM = 'roguard-three-component-review-unanimity-research-1'


def _validate_components(original, readout, attention, language, *, boundary=24, layers=42):
    """Reject incompatible tensors before sharing any frozen encoder state."""
    import numpy as np
    allowed_attention = {
        f'base.model.layers.{i}.self_attn.{projection}.lora_{factor}'
        for i in range(boundary, layers) for projection in ('q_proj', 'o_proj') for factor in ('a', 'b')}
    active_head = 'heads.'+language+'.'
    for kind, candidate in (('readout', readout), ('attention', attention)):
        if set(candidate) != set(original):
            raise ValueError('Committee component tensor inventories differ')
        for name, value in candidate.items():
            if value.shape != original[name].shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid committee component tensor: '+name)
            permitted = name.startswith(active_head) or (kind == 'attention' and name in allowed_attention)
            if not permitted and not np.array_equal(value, original[name]):
                raise ValueError('Committee component changes the shared encoder or inactive head: '+name)


class _CommitteeEngine:
    def __init__(self, args, specs):
        from safetensors.numpy import load
        tensors = {}
        for name in COMPONENTS:
            raw = _read(args[name][1], 16_000_000)
            if _sha(raw) != args[name][2]:
                raise ValueError('Committee component artifact identity differs')
            tensors[name] = load(raw)
        _validate_components(tensors['v9'], tensors['readout'], tensors['attention'], specs['v9']['language'])
        self.engine = _Engine(*args['v9'])
        self.tensors = tensors
        self.boundary = 24
        # Validate every inventory against the actual model, then leave v9 active.
        try:
            for name in COMPONENTS:
                _load_weights(self.engine.model, tensors[name])
        finally:
            _load_weights(self.engine.model, tensors['v9'])

    def scores(self, prefixes, specs):
        engine = self.engine; mx = engine.mx
        sequences = [classifier_tokens(engine.tokenizer, payload) for payload in prefixes]
        common = len(PREFIX_TOKENS); cache = None
        if any(common < len(s) <= 1024 and s[:common] == PREFIX_TOKENS for s in sequences):
            cache = engine.make_cache(engine.model.base)
            mx.eval(engine.model.base.model(mx.array([PREFIX_TOKENS]), cache=cache))
        output = []
        for sequence in sequences:
            if len(sequence) > 1024:
                output.append(None)
                continue
            if cache is not None and len(sequence) > common and sequence[:common] == PREFIX_TOKENS:
                state = capture_with_prefix(engine.model.base, mx.array([sequence[common:]]), cache, start=self.boundary)
            else:
                state = capture(engine.model.base, mx.array([sequence]), start=self.boundary)
            values = {}
            try:
                original = replay(engine.model.base, state)[:, 0, :]; mx.eval(original)
                for name in COMPONENTS:
                    if name != 'v9':
                        _load_weights(engine.model, self.tensors[name])
                    features = replay(engine.model.base, state)[:, 0, :] if name == 'attention' else original
                    logits = engine.model.heads[specs[name]['language']](features); mx.eval(logits)
                    score = _score(logits[0], specs[name])
                    values[name] = {'score': score, 'predicted': int(score >= specs[name]['threshold']),
                                    'field_logits': logits[0].tolist()}
                output.append(values)
            finally:
                # An exception must not leave an attention/readout overlay active.
                _load_weights(engine.model, self.tensors['v9'])
                mx.clear_cache()
            del state, original, features, logits
        return output


class Gemma4ReviewCommittee:
    """Explicit research committee over three checksum-bound local models.

    Unanimity can suppress a real concern if any component misses it. Results
    expose disagreement and confer no confidence or authority to take action.
    The v9 12-turn/1024-token input contract still applies to this component.
    """
    def __init__(self, *, v9, readout, attention):
        components = dict(v9=v9, readout=readout, attention=attention)
        if any(not isinstance(model, Gemma4ReviewModel) for model in components.values()):
            raise ValueError('Committee requires validated Gemma4 model configurations')
        if len({m.language for m in components.values()}) != 1 or len({m._args[0].resolve() for m in components.values()}) != 1:
            raise ValueError('Committee components must share a language and pinned base')
        self.language = v9.language
        self._specs = {name: deepcopy(m._spec) for name, m in components.items()}
        self._args = {name: m._args for name, m in components.items()}
        self.component_identities = {name: m.identity for name, m in components.items()}
        self.identity = digest({'algorithm': ALGORITHM, 'components': self.component_identities})
        self._engine = None

    def predict(self, payload, *, focus=None):
        return self.predict_many([payload], focuses=[focus])[0]

    def predict_many(self, payloads, *, focuses=None):
        if not isinstance(payloads, list) or not 1 <= len(payloads) <= 128:
            raise ValueError('Expected one to 128 conversations')
        if focuses is None:
            focuses = [None]*len(payloads)
        if not isinstance(focuses, list) or len(focuses) != len(payloads):
            raise ValueError('Focal indices do not match the batch')
        prepared = [prepare_input(p, f) for p, f in zip(payloads, focuses)]
        if any(p[0]['language'] != self.language for p in prepared):
            raise ValueError('Committee language differs')
        with _LOCK:
            if self._engine is None:
                self._engine = _CommitteeEngine(self._args, self._specs)
            values = self._engine.scores([p[0] for p in prepared], self._specs)
        if not isinstance(values, list) or len(values) != len(prepared):
            raise ValueError('Incomplete committee output')
        output = []
        for components, (_, focus, identity) in zip(values, prepared):
            if components is not None:
                if not isinstance(components, dict) or set(components) != set(COMPONENTS):
                    raise ValueError('Incomplete committee component output')
                for name, row in components.items():
                    score = _score(row['field_logits'], self._specs[name])
                    if (type(row['score']) not in (int, float) or not math.isfinite(row['score'])
                            or abs(score-row['score']) > 1e-7 or type(row['predicted']) is not int
                            or row['predicted'] != int(score >= self._specs[name]['threshold'])):
                        raise ValueError('Committee score and decision differ')
            predicted = None if components is None else int(all(components[k]['predicted'] == 1 for k in COMPONENTS))
            disagreement = None if components is None else len({r['predicted'] for r in components.values()}) > 1
            output.append({'format': ALGORITHM, 'language': self.language, 'focus': focus,
                'input_digest': identity, 'model_identity': self.identity,
                'component_identities': dict(self.component_identities), 'components': components,
                'component_disagreement': disagreement, 'predicted': predicted,
                'score': predicted, 'score_kind': 'binary_unanimity_indicator', 'score_is_calibrated': False,
                'decision': 'clarify' if predicted is None else 'review' if predicted else 'no_signal',
                'reason': 'INPUT_EXCEEDS_MODEL_CONTEXT' if predicted is None else 'UNANIMOUS_REVIEW_PROPOSAL' if predicted else 'NO_UNANIMOUS_REVIEW_PROPOSAL',
                'review_requested': predicted != 0, 'qualified': False, 'text_in_report': False,
                'establishes_safety': False, 'permits_external_action': False, 'permits_notification': False})
        return output
