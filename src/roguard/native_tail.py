"""Private native-tail80 serving prototype. Injected closure is a trust boundary.
No closure approval, export, calibration, package integration or qualification.
"""
from collections import namedtuple
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import threading
import traceback

FORMAT = 'roguard-native-tail19-manifest-2'
ROUTING = {'candidate_max_tokens': 636, 'original_max_tokens': 1024,
           'routes': ['native_tail', 'original_v9', 'original_v9_unsupported']}
PROVENANCE = {'selected_checkpoint_sha256', 'training_plan_sha256',
              'training_report_sha256'}
ASSETS = {'manifest', 'native_spec', 'native_adapter', 'spec9', 'adapter9'}
SOURCES = {'runtime', 'gemma4_review', 'gemma4_attention', 'gemma4_contract'}
SPEC9 = {'ro': '6b2ea2979ffa43c735d8f8a6e3f7e1fbf6cf5c085889e8d484a13bf43026ccc5',
         'uk': 'fc86fed9981b786149c4f6855c3d98f07067654cf2af05d40aef589345be34ed'}
WEIGHTS9 = {'ro': '114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
            'uk': '0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f'}
DetailedPrediction = namedtuple('DetailedPrediction', 'response logits route spec_sha256')
_LEASE_LOCK = threading.RLock()
_OWNER = None  # Strong reference: abandonment cannot grant a second encoder.


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_sha(x):
    return type(x) is str and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def tensor_schema():
    """Pinned Gemma4 E4B geometry, rank8 q/o layers24..41, both128x19 heads."""
    result = {}
    for layer in range(24, 42):
        width = 4096 if layer % 6 == 5 else 2048
        prefix = f'base.model.layers.{layer}.self_attn.'
        for name, shape in [('q_proj.lora_a', [2560, 8]), ('q_proj.lora_b', [8, width]),
                            ('o_proj.lora_a', [width, 8]), ('o_proj.lora_b', [8, 2560])]:
            result[prefix + name] = shape
    for lang in ('ro', 'uk'):
        for name, shape in [('first.weight', [128, 2560]), ('first.bias', [128]),
                            ('last.weight', [19, 128]), ('last.bias', [19])]:
            result[f'heads.{lang}.{name}'] = shape
    return result


def checked_tensors(raw, inventory=None):
    import numpy as np
    from safetensors.numpy import load
    values = load(raw)
    schema = tensor_schema()
    require(set(values) == set(schema), 'Exact80 tensor inventory required')
    if inventory is not None:
        require(type(inventory) is dict and set(inventory) == set(schema), 'Manifest tensor keys')
    result = {}
    for name, shape in schema.items():
        x = values[name]
        require(x.dtype == np.float32 and list(x.shape) == shape and np.isfinite(x).all(), 'Invalid tensor ' + name)
        proof = dict(dtype='float32', shape=shape, sha256=digest(x.tobytes(order='C')))
        if inventory is not None:
            require(inventory[name] == proof, 'Tensor proof mismatch ' + name)
        # Immutable backing bytes, not merely an ndarray writeable flag.
        result[name] = np.frombuffer(x.tobytes(order='C'), dtype=np.float32).reshape(shape)
    return result


def source_digests():
    from roguard import gemma4_review, gemma4_attention, gemma4_contract
    return {k: digest(Path(p).read_bytes()) for k, p in
            [('runtime', __file__), ('gemma4_review', gemma4_review.__file__),
             ('gemma4_attention', gemma4_attention.__file__), ('gemma4_contract', gemma4_contract.__file__)]}


def load(*, enabled, configured_parent=None, closure=None, expected_closure_identity=None):
    """Disabled returns the configured parent itself, without touching candidate assets.

    Enabled closure contract: externally registered identity; verify(); asset_path(name),
    asset_sha(name); base_path(). It owns finite path admission, all base/dependency/native
    bytes and actual installed source origins. This module does not mint that approval.
    """
    require(type(enabled) is bool, 'Explicit enabled boolean required')
    if not enabled:
        require(configured_parent is not None, 'Disabled parent required')
        return configured_parent
    require(closure is not None and valid_sha(expected_closure_identity), 'Registered closure required')
    closure.verify()
    require(closure.identity == expected_closure_identity, 'Closure identity mismatch')
    raw = {}
    for key in sorted(ASSETS):
        path = closure.asset_path(key)
        require(Path(path).stat().st_size <= 16_000_000, 'Bounded asset required')
        raw[key] = Path(path).read_bytes()
        require(digest(raw[key]) == closure.asset_sha(key), 'Asset digest mismatch ' + key)
    m = strict_json(raw['manifest'])
    keys = {'format', 'qualified', 'language', 'native_spec_sha256', 'adapter_sha256',
            'original_v9_spec_sha256', 'original_v9_adapter_sha256', 'tensor_inventory',
            'runtime_sources', 'routing', 'provenance'}
    require(type(m) is dict and set(m) == keys and m['format'] == FORMAT and m['qualified'] is False,
            'Unsupported native manifest')
    lang = m['language']
    require(lang in SPEC9 and m['routing'] == ROUTING, 'Routing/language mismatch')
    require(type(m['provenance']) is dict and set(m['provenance']) == PROVENANCE
            and all(valid_sha(v) for v in m['provenance'].values()), 'Exact provenance digests required')
    require(m['provenance']['selected_checkpoint_sha256'] == m['adapter_sha256'],
            'Selected full80 checkpoint must be the exact served adapter bytes')
    require(type(m['runtime_sources']) is dict and set(m['runtime_sources']) == SOURCES
            and m['runtime_sources'] == source_digests(), 'Runtime source mismatch')
    for key, field in [('native_spec', 'native_spec_sha256'), ('native_adapter', 'adapter_sha256'),
                       ('spec9', 'original_v9_spec_sha256'), ('adapter9', 'original_v9_adapter_sha256')]:
        require(digest(raw[key]) == m[field], 'Manifest asset edge mismatch')
    require(m['original_v9_spec_sha256'] == SPEC9[lang] and m['original_v9_adapter_sha256'] == WEIGHTS9[lang],
            'Exact original-v9 parent required')
    original_spec, candidate_spec = strict_json(raw['spec9']), strict_json(raw['native_spec'])
    require(candidate_spec == dict(original_spec, adapter_sha256=m['adapter_sha256']),
            'Candidate must preserve own-v9 policy/threshold and all native spec fields')
    require(original_spec['adapter_sha256'] == m['original_v9_adapter_sha256'], 'Original adapter edge')
    original = checked_tensors(raw['adapter9'])
    candidate = checked_tensors(raw['native_adapter'], m['tensor_inventory'])
    inactive = 'uk' if lang == 'ro' else 'ro'
    require(all(candidate[k].tobytes() == original[k].tobytes() for k in candidate
                if k.startswith('heads.' + inactive + '.')), 'Inactive head changed')
    from roguard.gemma4_review import Gemma4ReviewModel
    base = closure.base_path()
    parents = {route: Gemma4ReviewModel(closure.asset_path(spec), closure.asset_path(adapter), base,
                                       expected_sha256=m[field])
               for route, spec, adapter, field in
               [('native_tail', 'native_spec', 'native_adapter', 'native_spec_sha256'),
                ('original_v9', 'spec9', 'adapter9', 'original_v9_spec_sha256')]}
    closure.verify()
    require(closure.identity == expected_closure_identity, 'Closing closure identity mismatch')
    return NativeTail(parents, candidate, original, closure, expected_closure_identity)


class _Observer:
    def __init__(self, head, mx):
        self.head, self.mx, self.captured = head, mx, []
    def __getattr__(self, name):
        return getattr(self.head, name)
    def __call__(self, features):
        import numpy as np
        result = self.head(features)
        self.mx.eval(result)
        a = np.asarray(result)
        require(a.dtype == np.float32 and a.shape == (1, 19) and np.isfinite(a).all(), 'Native head output invalid')
        self.captured.append(tuple(float(x) for x in a[0]))
        return result  # The public scorer receives the exact original native object.


def _clear_error_frames(error, seen=None):
    # Tracebacks can retain capture/base/Metal state after engine aliases are cleared.
    if seen is None:
        seen = set()
    if id(error) in seen:
        return
    seen.add(id(error))
    traceback.clear_frames(error.__traceback__)
    for child in [error.__cause__, error.__context__, *getattr(error, 'exceptions', ())]:
        if child is not None:
            _clear_error_frames(child, seen)


class NativeTail:
    def __init__(self, parents, candidate, original, closure, identity):
        self._parents, self._candidate, self._original = parents, candidate, original
        self._closure, self.closure_identity = closure, identity
        self.language = parents['native_tail'].language
        self.identity = parents['native_tail'].identity
        self._engine, self._closed, self._poisoned = None, False, False
        self._parent_proofs = {k: self._parent_proof(p) for k, p in parents.items()}
        self._tensor_proofs = {route: {k: (v.dtype.str, tuple(v.shape), digest(v.tobytes())) for k, v in data.items()}
                               for route, data in [('native_tail', candidate), ('original_v9', original)]}
    @staticmethod
    def _parent_proof(parent):
        return (parent.identity, parent.language, tuple(map(str, parent._args)),
                json.dumps(parent._spec, sort_keys=True, separators=(',', ':'), allow_nan=False))
    def _verify(self):
        require(not self._closed, 'Native tail is closed')
        self._closure.verify()
        require(self._closure.identity == self.closure_identity, 'Closure identity changed')
        require(all(p._engine is self._engine for p in self._parents.values()), 'Parent engine alias changed')
        require({k:self._parent_proof(p) for k,p in self._parents.items()} == self._parent_proofs,
                'Parent runtime configuration changed')
        for route, data in [('native_tail', self._candidate), ('original_v9', self._original)]:
            require(set(data) == set(tensor_schema()) and
                    {k:(v.dtype.str, tuple(v.shape), digest(v.tobytes())) for k,v in data.items()} == self._tensor_proofs[route],
                    'Retained tensor identity changed')
    def _acquire(self):
        global _OWNER
        from roguard import gemma4_review as g
        with _LEASE_LOCK:
            require(_OWNER is None or _OWNER is self, 'Another native-tail encoder lease is active')
            _OWNER = self
            if self._engine is None:
                try:
                    self._engine = g._Engine(*self._parents['original_v9']._args)
                    for parent in self._parents.values():
                        parent._engine = self._engine
                except BaseException:
                    # Constructor cleanup cannot be certified by this adapter.
                    self._poisoned = self._closed = True
                    raise
    def predict(self, payload, *, focus=None):
        return self.predict_many([payload], focuses=[focus])[0]
    def predict_many(self, payloads, *, focuses=None):
        return [x.response for x in self.predict_detailed_many(payloads, focuses=focuses)]
    def predict_detailed(self, payload, *, focus=None):
        return self.predict_detailed_many([payload], focuses=[focus])[0]
    def predict_detailed_many(self, payloads, *, focuses=None):
        from roguard import gemma4_review as g
        require(type(payloads) is list and 1 <= len(payloads) <= 128, 'Expected1..128 payloads')
        if focuses is None:
            focuses = [None] * len(payloads)
        require(type(focuses) is list and len(focuses) == len(payloads), 'Focus batch mismatch')
        payloads, focuses = deepcopy(payloads), deepcopy(focuses)
        prepared = [g.prepare_input(p, f) for p, f in zip(payloads, focuses)]
        require(all(p[0]['language'] == self.language for p in prepared), 'Language mismatch')
        with g._LOCK:
            results = [None] * len(payloads)
            try:
                self._verify()
                self._acquire()
                counts = [len(g.classifier_tokens(self._engine.tokenizer, p[0])) for p in prepared]
                # Two stable groups at most. Each public batch builds its own prefix cache;
                # it is destroyed before changing all80 tensors for the next route.
                for route in ('native_tail', 'original_v9'):
                    indices = [i for i,n in enumerate(counts) if ('native_tail' if n <= 636 else 'original_v9') == route]
                    if not indices:
                        continue
                    parent = self._parents[route]
                    g._load_weights(self._engine.model, self._candidate if route == 'native_tail' else self._original)
                    head = self._engine.model.heads[self.language]
                    observer = _Observer(head, self._engine.mx)
                    self._engine.model.heads[self.language] = observer
                    try:
                        responses = parent.predict_many([payloads[i] for i in indices],
                                                        focuses=[focuses[i] for i in indices])
                    finally:
                        self._engine.model.heads[self.language] = head
                    require(type(responses) is list and len(responses) == len(indices),
                            'Exact public batch result cardinality required')
                    require(len(observer.captured) == sum(counts[i] <= 1024 for i in indices),
                            'Exactly one head call per supported row required')
                    captured = iter(observer.captured)
                    for i, response in zip(indices, responses):
                        unsupported = counts[i] > 1024
                        require(response['model_identity'] == parent.identity and
                                response['input_digest'] == prepared[i][2] and response['focus'] == prepared[i][1],
                                'Public input/model identity mismatch')
                        require((response['predicted'] is None) == unsupported, 'Public support status mismatch')
                        results[i] = DetailedPrediction(response, None if unsupported else next(captured),
                                        'original_v9_unsupported' if unsupported else route, parent.identity)
                require(all(x is not None for x in results), 'Incomplete routed batch')
                g._load_weights(self._engine.model, self._original)
                self._verify()
                return results
            except BaseException as error:
                # Errors never route to a different model or become abstentions.
                _clear_error_frames(error)
                try:
                    self.close()
                except BaseException as cleanup:
                    raise BaseExceptionGroup('Inference and cleanup failed', [error, cleanup])
                raise
    def structured_fields(self, detail):
        route = 'native_tail' if detail.route == 'native_tail' else 'original_v9'
        require(detail.route in ROUTING['routes'] and detail.spec_sha256 == self._parents[route].identity,
                'Detailed route/spec identity mismatch')
        require(self._parent_proof(self._parents[route]) == self._parent_proofs[route], 'Own-spec mutated')
        return structured_fields(detail, self._parents[route]._spec)
    def close(self):
        global _OWNER
        from roguard.gemma4_review import _LOCK
        with _LOCK, _LEASE_LOCK:
            self._closed = True
            engine, self._engine = self._engine, None
            for parent in self._parents.values():
                parent._engine = None
            self._candidate = self._original = None
            if engine is not None:
                try:
                    engine.model = engine.tokenizer = engine.make_cache = None
                    gc.collect()
                finally:
                    try:
                        engine.mx.clear_cache()
                    except BaseException:
                        self._poisoned = True
                        raise
            if _OWNER is self and not self._poisoned:
                _OWNER = None


def structured_fields(detail, spec):
    """Same-call point fields + semantic432 distribution; no calibration/coverage claim.
    UK semantic probability is deliberately not substituted for legacy binary score.
    """
    import numpy as np
    from roguard.gemma4_review import _score
    from roguard.native_fields import FIELDS, SIZES
    from roguard.trajectory_neural import factors_from_logits
    from roguard.trajectory_sets import WORLDS
    if detail.logits is None:
        require(detail.route == 'original_v9_unsupported' and detail.response['predicted'] is None,
                'Unsupported route metadata mismatch')
        return dict(field_argmax=None, probabilities=None, conservative_worlds=deepcopy(WORLDS), qualified=False)
    z = np.asarray(detail.logits, dtype=np.float32)
    require(z.shape == (19,) and np.isfinite(z).all(), 'Invalid detailed logits')
    score = _score(z, spec)
    require(score.hex() == float(detail.response['score']).hex(), 'Own-spec score mismatch')
    require(detail.response['predicted'] == int(score >= spec['threshold']), 'Own-spec decision mismatch')
    fields, offset = {}, 0
    for name, size in zip(FIELDS, SIZES):
        fields[name] = int(np.argmax(z[offset:offset + size])); offset += size
    probabilities = factors_from_logits(z)
    return dict(field_argmax=fields, probabilities=probabilities, conservative_worlds=None,
                binary_decision=detail.response['predicted'], semantic_probability_is_legacy_score=spec['mode'] != 'binary',
                qualified=False)
