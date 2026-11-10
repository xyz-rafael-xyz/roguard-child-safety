"""Explicit local Gemma4 adapter review; no default selection or external actions."""
from copy import deepcopy
import hashlib
from importlib.metadata import distribution, version
import json
import math
from pathlib import Path
import tempfile
import threading

from .assurance import policy_digest
from .gemma4_contract import BASE_MODEL, BASE_REVISION, FILES, IMPLEMENTATIONS, POLICY_SHA256, PREFIX_TOKENS, VERSIONS, classifier_tokens
from .interpreter import prepare_input
from .language_bundle import _read_manifest
from .native_fields import FIELDS, SIZES, _check_couplings, factorized_review_mass
from .native_review import _digest, _read, _sha

_LOCK = threading.RLock()
LORA = dict(rank=8, scale=16.0, dropout=0.0, keys=['self_attn.q_proj', 'self_attn.o_proj'])
FORMAT = 'roguard-gemma4-tail-review-experiment-v1'


def _snapshot(base, destination):
    """Loaders see only the bytes copied and verified against the built-in pin."""
    destination.mkdir()
    for name, expected in FILES.items():
        total, hashed = 0, hashlib.sha256()
        with (Path(base)/name).open('rb') as source, (destination/name).open('xb') as target:
            while True:
                block = source.read(1024*1024)
                if not block:
                    break
                total += len(block)
                if total > expected['bytes']:
                    raise ValueError('Gemma4 base asset exceeds pinned size: ' + name)
                hashed.update(block)
                target.write(block)
        if total != expected['bytes'] or hashed.hexdigest() != expected['sha256']:
            raise ValueError('Gemma4 base identity differs: ' + name)


def _make_model(base, *, boundary=24, width=128, lora=None):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from .gemma4_attention import _backbone
    _backbone(base, boundary)
    base.freeze()
    linear_to_lora_layers(base, len(base.layers)-boundary, lora or LORA)
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(base.args.hidden_size, width)
            self.last = nn.Linear(width, sum(SIZES)+2)
        def __call__(self, features):
            values = features/mx.sqrt(mx.sum(mx.square(features), axis=1, keepdims=True))*features.shape[1]**.5
            return self.last(nn.gelu(self.first(values)))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.heads = {language: Head() for language in ('ro', 'uk')}
    return Model()


def _load_weights(model, tensors):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(tensors) != set(expected):
        raise ValueError('Gemma4 adapter/head tensor inventory differs')
    for name, value in tensors.items():
        if value.shape != expected[name].shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid Gemma4 adapter/head tensor: ' + name)
    model.load_weights([(name, mx.array(value)) for name, value in tensors.items()], strict=False)
    mx.eval(model.trainable_parameters())
    model.eval()


def _score(logits, spec):
    """Independent portable policy evaluation with no training imports."""
    import numpy as np
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (sum(SIZES)+2,) or not np.isfinite(values).all():
        raise ValueError('Invalid Gemma4 review logits')
    def probabilities(v):
        p = np.exp(v-v.max())
        return (p/p.sum()).tolist()
    if spec['mode'] == 'binary':
        return probabilities(values[-2:])[1]
    factors, offset = {}, 0
    for field, size in zip(FIELDS, SIZES):
        factors[field] = probabilities(values[offset:offset+size])
        offset += size
    return factorized_review_mass(factors, couplings=spec['couplings'] if spec['mode'] == 'coupled_policy' else None)


class _Engine:
    def __init__(self, base, adapter, expected):
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
            from safetensors.numpy import load as load_tensors
        except ImportError as error:
            raise ImportError('Gemma4 review requires Apple Silicon, Python 3.12 or later, and the gemma4 extra') from error
        for name, pinned in VERSIONS.items():
            if version(name) != pinned:
                raise ValueError('Gemma4 qualification requires '+name+'=='+pinned)
        package = Path(distribution('mlx-lm').locate_file('mlx_lm'))
        for name, expected_file in IMPLEMENTATIONS.items():
            if _sha((package/name).read_bytes()) != expected_file:
                raise ValueError('Gemma4 runtime implementation differs: ' + name)
        raw = _read(adapter, 16_000_000)
        if _sha(raw) != expected:
            raise ValueError('Gemma4 adapter identity differs')
        tensors = load_tensors(raw)
        self.mx, self.make_cache = mx, make_prompt_cache
        mx.set_memory_limit(10*1024**3)
        mx.set_cache_limit(256*1024**2)
        with tempfile.TemporaryDirectory(prefix='roguard-gemma4-') as temporary:
            path = Path(temporary)/'base'
            _snapshot(base, path)
            original, self.tokenizer = load(str(path), tokenizer_config={'trust_remote_code': False}, lazy=False)
            original.set_dtype(mx.float32)
            self.model = _make_model(original.language_model)
            _load_weights(self.model, tensors)
            mx.eval(self.model.parameters())
        del original, tensors, raw

    def scores(self, prefixes, spec):
        from .gemma4_attention import capture, capture_with_prefix, replay
        mx = self.mx
        sequences = [classifier_tokens(self.tokenizer, p) for p in prefixes]
        # Match the qualified extraction path, including its fixed prefix length.
        # This prefix contains only shared prompt scaffolding, not message text.
        common = len(PREFIX_TOKENS)
        cache = None
        if any(common < len(s) <= 1024 and s[:common] == PREFIX_TOKENS for s in sequences):
            cache = self.make_cache(self.model.base)
            mx.eval(self.model.base.model(mx.array([PREFIX_TOKENS]), cache=cache))
        output = []
        for sequence in sequences:
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
            output.append(_score(logits[0], spec))
            del state, features, logits
            mx.clear_cache()
        # No conversation cache survives this method call.
        return output


class Gemma4ReviewModel:
    """Hash-bound experimental model. This class does not qualify it for release."""
    def __init__(self, artifact, adapter, base, *, expected_sha256):
        spec = _read_manifest(Path(artifact), expected_sha256)
        required = {'format', 'language', 'base_model', 'base_revision', 'policy_sha256',
                    'policy_digest', 'adapter_sha256', 'mode', 'couplings', 'threshold', 'boundary', 'lora'}
        if (not isinstance(spec, dict) or set(spec) != required or spec['format'] != FORMAT
                or spec['language'] not in ('ro', 'uk')
                or spec['base_model'] != BASE_MODEL or spec['base_revision'] != BASE_REVISION
                or spec['policy_sha256'] != POLICY_SHA256 or spec['policy_digest'] != policy_digest()
                or type(spec['boundary']) is not int or spec['boundary'] != 24 or spec['lora'] != LORA
                or spec['mode'] not in ('binary', 'policy', 'coupled_policy')
                or type(spec['threshold']) not in (int, float) or not math.isfinite(spec['threshold'])
                or not 0 <= spec['threshold'] <= 1):
            raise ValueError('Unsupported Gemma4 review contract')
        _check_couplings(spec['couplings'])
        if spec['mode'] == 'coupled_policy' and spec['couplings'] is None:
            raise ValueError('Selected coupled policy requires its learned tables')
        self._spec = deepcopy(spec)
        self._args = (Path(base), Path(adapter), _digest(spec['adapter_sha256']))
        self.identity, self.language, self._engine = expected_sha256, spec['language'], None

    @classmethod
    def from_bundle(cls, path, *, expected_sha256):
        from .field_pipeline import _local_path
        path = Path(path).resolve()
        spec = _read_manifest(path, expected_sha256)
        if (not isinstance(spec, dict) or set(spec) != {'format', 'artifact_sha256', 'assets'}
                or spec['format'] != 'roguard-gemma4-tail-review-bundle-v1'
                or not isinstance(spec['assets'], dict) or set(spec['assets']) != {'artifact', 'adapter', 'base'}):
            raise ValueError('Unsupported Gemma4 review bundle')
        assets = {key: _local_path(path.parent, value) for key, value in spec['assets'].items()}
        return cls(**assets, expected_sha256=_digest(spec['artifact_sha256']))

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
        if any(prefix['language'] != self.language for prefix, _, _ in prepared):
            raise ValueError('Gemma4 adapter language differs')
        with _LOCK:
            if self._engine is None:
                self._engine = _Engine(*self._args)
            scores = self._engine.scores([p[0] for p in prepared], self._spec)
        if not isinstance(scores, list) or len(scores) != len(prepared):
            raise ValueError('Incomplete Gemma4 review output')
        results = []
        for score, (prefix, focus, identity) in zip(scores, prepared):
            if score is not None and (type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('Invalid Gemma4 review score')
            predicted = None if score is None else int(score >= self._spec['threshold'])
            results.append(dict(format='roguard-gemma4-tail-review-v1', language=self.language, focus=focus,
                input_digest=identity, model_identity=self.identity, score=score, predicted=predicted,
                decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',
                reason='INPUT_EXCEEDS_MODEL_CONTEXT' if predicted is None else 'MODEL_REVIEW_PROPOSAL' if predicted else 'NO_SIGNAL_DETECTED',
                review_requested=predicted != 0, score_is_calibrated=False, text_in_report=False,
                establishes_safety=False, permits_external_action=False, permits_notification=False))
        return results
