"""Research response attention adapters sharing a validated local review encoder."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import unicodedata

from .assurance import digest
from .gemma4_contract import BASE_MODEL, BASE_REVISION
from .gemma4_response import LABELS, QUERY_SHA256, _probabilities, _without_adapters, response_tokens
from .gemma4_review import Gemma4ReviewModel, _Engine, _LOCK
from .language_bundle import _read_manifest
from .native_review import _read, _sha

FORMAT = 'roguard-gemma4-response-attention-research-1'


def _adapter_slots(base):
    from .gemma4_attention import _backbone
    body = _backbone(base, 24)
    slots = {}
    for index, layer in enumerate(body.layers[24:], 24):
        for projection in ('q_proj', 'o_proj'):
            module = getattr(layer.self_attn, projection)
            for factor in ('a', 'b'):
                attribute = 'lora_'+factor
                if not hasattr(module, attribute):
                    raise ValueError('Missing response attention adapter')
                slots[f'base.model.layers.{index}.self_attn.{projection}.{attribute}'] = (module, attribute)
    if len(slots) != 72:
        raise ValueError('Unexpected response attention inventory')
    return slots


@contextmanager
def _replace_adapters(base, tensors):
    slots = _adapter_slots(base)
    if set(tensors) != set(slots):
        raise ValueError('Response attention replacement inventory differs')
    saved = {key: getattr(module, attribute) for key, (module, attribute) in slots.items()}
    if any(tensors[key].shape != saved[key].shape for key in slots):
        raise ValueError('Response attention replacement shape differs')
    try:
        for key, (module, attribute) in slots.items():
            setattr(module, attribute, tensors[key])
        yield
    finally:
        for key, (module, attribute) in slots.items():
            setattr(module, attribute, saved[key])


def _features(engine, payload):
    """The prompt cache ends in frozen shared-KV layers; replay uses active adapters."""
    from .gemma4_attention import capture, capture_with_prefix, replay
    import numpy as np
    mx = engine.mx
    tokens = response_tokens(engine.tokenizer, payload)
    if not tokens or len(tokens) > 1024:
        return None
    cached = getattr(engine, '_response_prefix', None)
    if cached is None:
        probes = [response_tokens(engine.tokenizer, {'turns': turns}) for turns in (
            [{'role': 'minor', 'text': 'x'}],
            [{'role': 'minor', 'text': 'y'}, {'role': 'minor', 'text': 'z'}])]
        common = 0
        for values in zip(*probes):
            if len(set(values)) != 1:
                break
            common += 1
        if not common or common >= min(map(len, probes)):
            raise ValueError('Invalid response query prefix')
        prefix_tokens = probes[0][:common]
        prefix = engine.make_cache(engine.model.base)
        with _without_adapters(engine.model.base):
            mx.eval(engine.model.base.model(mx.array([prefix_tokens]), cache=prefix))
        cached = (QUERY_SHA256, prefix_tokens, prefix)
        engine._response_prefix = cached
    query_sha, prefix_tokens, prefix = cached
    if query_sha != QUERY_SHA256:
        raise ValueError('Cached response query differs')
    common = len(prefix_tokens)
    if len(tokens) > common and tokens[:common] == prefix_tokens:
        state = capture_with_prefix(engine.model.base, mx.array([tokens[common:]]), prefix, start=24)
    else:
        state = capture(engine.model.base, mx.array([tokens]), start=24)
    hidden = replay(engine.model.base, state)[:, 0, :]
    mx.eval(hidden)
    values = np.asarray(hidden).copy()
    if values.shape != (1, 2560) or not np.isfinite(values).all():
        raise ValueError('Invalid response attention representation')
    return values


class _AdapterReadout:
    def __init__(self, path, expected, base):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from safetensors.numpy import load
        raw = _read(path, 16_000_000)
        if _sha(raw) != expected:
            raise ValueError('Response attention weights identity differs')
        tensors = load(raw)
        slots = _adapter_slots(base)
        shapes = {key: getattr(module, attribute).shape for key, (module, attribute) in slots.items()}
        shapes.update({'head.last.weight': (5, 2560), 'head.last.bias': (5,)})
        if set(tensors) != set(shapes):
            raise ValueError('Response attention checkpoint inventory differs')
        if any(value.shape != shapes[key] or value.dtype != np.float32 or not np.isfinite(value).all()
               for key, value in tensors.items()):
            raise ValueError('Invalid response attention checkpoint tensor')
        self.adapters = {key: mx.array(tensors[key]) for key in slots}
        self.head = nn.Linear(2560, 5)
        self.head.weight = mx.array(tensors['head.last.weight'])
        self.head.bias = mx.array(tensors['head.last.bias'])
        mx.eval(self.adapters, self.head.parameters())
        self.head.eval()

    def logits(self, features):
        import mlx.core as mx
        import numpy as np
        values = mx.array(features)
        values = values/mx.sqrt(mx.maximum(mx.mean(mx.square(values), axis=1, keepdims=True), 1e-12))
        values = self.head(values)
        mx.eval(values)
        return np.asarray(values)[0].tolist()


class Gemma4ResponseAttentionModel:
    """Unqualified bilingual response adapter; no downloads or external actions.

    Uses the caller's encoder under the review lock and restores all 72 original
    adapter arrays after both successful and failed calls. The binary head is
    never replaced. Probabilities are uncalibrated diagnostic scores.
    """
    def __init__(self, artifact, weights, reference, *, expected_sha256):
        if not isinstance(reference, Gemma4ReviewModel):
            raise ValueError('Expected a validated local Gemma4 reference')
        spec = _read_manifest(Path(artifact), expected_sha256)
        required = {'format', 'base_model', 'base_revision', 'query_sha256', 'adapter_sha256',
                    'boundary', 'hidden', 'labels', 'qualified'}
        if (not isinstance(spec, dict) or set(spec) != required or spec['format'] != FORMAT
                or spec['base_model'] != BASE_MODEL or spec['base_revision'] != BASE_REVISION
                or spec['query_sha256'] != QUERY_SHA256 or spec['labels'] != list(LABELS)
                or type(spec['boundary']) is not int or spec['boundary'] != 24
                or type(spec['hidden']) is not int or spec['hidden'] != 0 or spec['qualified'] is not False
                or not isinstance(spec['adapter_sha256'], str) or len(spec['adapter_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in spec['adapter_sha256'])):
            raise ValueError('Invalid experimental response attention specification')
        self._spec = deepcopy(spec)
        self.reference = reference
        self._reference_binding = (reference.identity, reference._args, digest(reference._spec))
        self._weights = Path(weights)
        self._readout = None
        self.identity = digest({'format': FORMAT, 'artifact': expected_sha256, 'reference': reference.identity})

    def review_response(self, draft, language, context=''):
        if (language not in ('ro', 'uk') or not isinstance(draft, str) or not 1 <= len(draft.strip()) <= 6000
                or len(draft) > 6000 or not isinstance(context, str) or len(context) > 12000):
            raise ValueError('Expected Romanian/Ukrainian draft and bounded context')
        draft.encode('utf-8'); context.encode('utf-8')
        draft = unicodedata.normalize('NFC', draft); context = unicodedata.normalize('NFC', context)
        turns = ([{'role': 'minor', 'text': context}] if context else [])+[{'role': 'assistant', 'text': draft}]
        payload = {'language': language, 'turns': turns}
        with _LOCK:
            if (self.reference.identity, self.reference._args, digest(self.reference._spec)) != self._reference_binding:
                raise ValueError('Reference configuration changed')
            if self.reference._engine is None:
                self.reference._engine = _Engine(*self.reference._args)
            engine = self.reference._engine
            if self._readout is None:
                self._readout = _AdapterReadout(self._weights, self._spec['adapter_sha256'], engine.model.base)
            try:
                with _replace_adapters(engine.model.base, self._readout.adapters):
                    features = _features(engine, payload)
                    probabilities = None if features is None else _probabilities(self._readout.logits(features))
            finally:
                engine.mx.clear_cache()
        label = None if probabilities is None else LABELS[max(range(5), key=probabilities.__getitem__)]
        return {'format': FORMAT, 'language': language, 'model_identity': self.identity,
                'input_digest': digest(payload), 'label': label,
                'probabilities': None if probabilities is None else dict(zip(LABELS, probabilities)),
                'probabilities_calibrated': False, 'abstained': label is None,
                'reason': 'INPUT_EXCEEDS_MODEL_CONTEXT' if label is None else 'SINGLE_CATEGORY_RESPONSE_DIAGNOSTIC',
                'review_requested': label != 'appropriate', 'text_in_report': False,
                'qualified': False, 'safe_response_established': False,
                'contextual_factual_support_established': False,
                'permits_external_action': False, 'permits_notification': False}
