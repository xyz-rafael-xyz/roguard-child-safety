"""Optional experimental role residual runtime; no qualified artifact is supplied.

Disabled mode delegates the parent's original scoring method without loading
numerical extras, alignment or residual weights. Enabled mode is experimental.
"""
import hashlib, json, math, re, sys
from copy import deepcopy
from pathlib import Path
SHAPES = {'first.weight': (32, 5638), 'first.bias': (32,), 'last.weight': (17, 32), 'last.bias': (17,)}
ARCHITECTURE = 'qualified-q2560-role-value23-3x1024-metadata6-residual32-semantic17-v1'
NORMALIZATION = {'method': 'per-block-l2-to-sqrt-width', 'norm_dtype': 'float64', 'output_dtype': 'float32', 'minimum_norm': 1e-06, 'count_scale': 1024, 'gelu': 'exact-erf', 'legacy': 'unchanged-last2'}
ALIGNMENT_SHA = '0c65cfcb800a445c52e9c880475e485ea9d35e8241468c92b7f861c37658ad6a'
ALIGNMENT_LINEAGE_SHA = 'd684b8b592eed2e5241945211bf689097ca09a3bb44b58eb4dba3b7650bdbc5c'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def bounded(path, limit):
    with Path(path).open('rb') as f:
        raw = f.read(limit + 1)
    require(len(raw) <= limit, 'Asset exceeds fixed reader limit')
    return raw

def hash_value(x):
    return isinstance(x, str) and re.fullmatch('[0-9a-f]{64}', x) is not None

def validate_manifest(value):
    required = {'format', 'language', 'architecture', 'tensor_sha256', 'parent', 'normalization', 'alignment_source_sha256', 'alignment_lineage_sha256', 'runtime_source_sha256', 'maximum_tokens', 'fallback', 'provenance', 'qualified'}
    require(isinstance(value, dict) and set(value) == required, 'Residual manifest inventory differs')
    require(value['format'] == 'roguard-role-residual-experiment-1' and value['architecture'] == ARCHITECTURE and (value['language'] in ('ro', 'uk')) and (value['qualified'] is False), 'Only explicit unqualified research artifact supported')
    require(value['runtime_source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'Packaged residual runtime source differs')
    require(digest(value['normalization']) == digest(NORMALIZATION) and value['alignment_source_sha256'] == ALIGNMENT_SHA and (value['alignment_lineage_sha256'] == ALIGNMENT_LINEAGE_SHA), 'Normalization/alignment contract differs')
    require(type(value['maximum_tokens']) is int and value['maximum_tokens'] == 613, 'Invalid registered residual envelope')
    require(digest(value['fallback']) == digest({'above_residual_envelope': 'original_v9', 'maximum_parent_tokens': 1024, 'on_error': False, 'on_low_confidence': False}), 'Explicit original-v9 fallback contract required')
    require(hash_value(value['tensor_sha256']), 'Residual hash required')
    require(set(value['parent']) == {'original_v9_adapter_sha256', 'original_v9_spec_sha256', 'head75_source_sha256', 'spec_sha256', 'base_revision'} and all((hash_value(value['parent'][k]) for k in ('original_v9_adapter_sha256', 'original_v9_spec_sha256', 'head75_source_sha256', 'spec_sha256'))), 'Parent identity inventory differs')
    require(value['parent']['base_revision'] == '475b9088d29754a3379866cf5aeb6b41acd313c2', 'Base revision differs')
    require(set(value['provenance']) == {'selected_checkpoint_sha256', 'training_plan_sha256', 'feature_manifest_sha256', 'selection_proof_sha256'} and all((hash_value(x) for x in value['provenance'].values())), 'Selected artifact provenance required')
    return deepcopy(value)

class ResidualArrays:

    def __init__(self, weights):
        import numpy as np
        require(set(weights) == set(SHAPES), 'Exact four residual tensors required')
        self._weights = {}
        for name, shape in SHAPES.items():
            v = weights[name]
            require(isinstance(v, np.ndarray) and v.shape == shape and (v.dtype == np.float32) and np.isfinite(v).all(), 'Invalid residual tensor: ' + name)
            self._weights[name] = v.copy()
            self._weights[name].flags.writeable = False

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    @classmethod
    def from_files(cls, manifest_path, tensor_path, *, expected_manifest_sha256):
        from safetensors.numpy import load
        raw = bounded(manifest_path, 128 * 1024)
        require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256, 'Residual manifest bytes differ')
        manifest = validate_manifest(json.loads(raw))
        tensor_raw = bounded(tensor_path, 1024 * 1024)
        require(hashlib.sha256(tensor_raw).hexdigest() == manifest['tensor_sha256'], 'Residual tensor bytes differ')
        result = cls(load(tensor_raw))
        result._manifest = manifest
        result.identity = expected_manifest_sha256
        return result

    def __call__(self, parent, q, roles, metadata):
        import numpy as np
        require(isinstance(parent, np.ndarray) and parent.ndim == 2, 'Batched parent logits required')
        n = len(parent)
        require(n > 0, 'Nonempty batch required')
        for v, shape in [(parent, (n, 19)), (q, (n, 2560)), (roles, (n, 3, 1024)), (metadata, (n, 6))]:
            require(isinstance(v, np.ndarray) and v.dtype == np.float32 and (v.shape == shape) and np.isfinite(v).all(), 'Invalid finite float32 residual input')
        counts, presence = (metadata[:, :3], metadata[:, 3:])
        require(np.all((counts >= 0) & (counts <= 1)) and np.all(counts * 1024 % 1 == 0), 'Invalid eligible counts')
        require(np.all((presence == 0) | (presence == 1)) and np.all((counts == 0) | (presence == 1)), 'Invalid structural presence')
        require(np.all(roles[counts == 0] == 0), 'Empty pool must be zero')
        require(np.all(np.square(q.astype(np.float64)).sum(-1) > 0), 'Final feature norm must be positive')

        def normalize(x):
            v = x.astype(np.float64)
            norm = np.maximum(np.sqrt(np.square(v).sum(-1, keepdims=True)), 1e-06)
            return (v / norm * math.sqrt(v.shape[-1])).astype(np.float32)
        x = np.concatenate((normalize(q), normalize(roles).reshape(n, 3072), metadata), -1)
        w = self._weights
        hidden = x @ w['first.weight'].T + w['first.bias']
        error_function = np.asarray([math.erf(float(v) / math.sqrt(2)) for v in hidden.reshape(-1)], dtype=np.float32).reshape(hidden.shape)
        hidden = hidden * (np.float32(1) + error_function) * np.float32(0.5)
        delta = hidden @ w['last.weight'].T + w['last.bias']
        require(np.isfinite(delta).all(), 'Nonfinite residual output')
        result = parent.copy()
        result[:, :17] = parent[:, :17] + delta
        require(np.isfinite(result).all(), 'Nonfinite combined logits')
        require(result[:, 17:].tobytes() == parent[:, 17:].tobytes(), 'Legacy logits changed')
        return result

def value_pools(value, positions, mx, np):
    require(value.ndim == 4 and value.shape[0] == 1 and (value.shape[1] == 2) and (value.shape[3] == 512), 'Global VALUE shape differs')
    require(len(positions) == 3 and all((p == sorted(set(p)) and all((type(i) is int and 387 <= i < value.shape[2] for i in p)) for p in positions)), 'Invalid global role coordinates')
    require(len(set(sum(positions, []))) == sum(map(len, positions)), 'Role masks overlap')
    values = value.astype(mx.float32).transpose(0, 2, 1, 3)[0]
    mx.eval(values)
    means = []
    for pos in positions:
        mean = mx.mean(values[mx.array(pos)], axis=0).reshape(1024) if pos else mx.zeros((1024,), dtype=mx.float32)
        mx.eval(mean)
        means.append(np.asarray(mean).copy())
    result = np.stack(means)
    require(np.isfinite(result).all(), 'Nonfinite VALUE means')
    return result

class BoundSpec:

    def __init__(self, path, expected_sha256):
        raw = bounded(path, 128 * 1024)
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'Parent spec bytes differ')
        self.value = json.loads(raw)
        self.sha256 = expected_sha256
        self.canonical = digest(self.value)
        from roguard.native_fields import _check_couplings
        require(self.value['language'] in ('ro', 'uk') and self.value['mode'] in ('binary', 'policy', 'coupled_policy') and (type(self.value['threshold']) in (int, float)) and math.isfinite(self.value['threshold']) and (0 <= self.value['threshold'] <= 1), 'Invalid parent policy spec')
        _check_couplings(self.value['couplings'])
        require(self.value['mode'] != 'coupled_policy' or self.value['couplings'] is not None, 'Coupled policy requires learned tables')

def render_result(score, spec, identity, model_identity, focus, format_name):
    require(score is None or (type(score) in (float, int) and math.isfinite(score) and (0 <= score <= 1)), 'Invalid serving score')
    predicted = None if score is None else int(score >= spec['threshold'])
    return dict(format=format_name, language=spec['language'], focus=focus, input_digest=identity, model_identity=model_identity, score=score, predicted=predicted, decision='clarify' if predicted is None else 'review' if predicted else 'no_signal', reason='INPUT_EXCEEDS_MODEL_CONTEXT' if predicted is None else 'MODEL_REVIEW_PROPOSAL' if predicted else 'NO_SIGNAL_DETECTED', review_requested=predicted != 0, score_is_calibrated=False, text_in_report=False, establishes_safety=False, permits_external_action=False, permits_notification=False)

def route_for_length(length, maximum_residual):
    require(type(length) is int and length > 387, 'Classifier sequence required')
    return 'residual' if length <= maximum_residual else 'original_v9' if length <= 1024 else 'original_v9_unsupported'

class OptionalResidualEngine:
    """One already-verified original-v9 backbone, frozen75 head, retained v9 head.
 Parent construction/install and original_head identity must be verified before
 attachment. Enabled prediction is route-aware; scores-only interface refuses
 mixed-threshold use. No extra backbone or failure/confidence fallback exists.
 """

    def __init__(self, parent, *, enabled, residual=None, alignment=None, parent_spec=None, original_spec=None, original_head=None):
        require(type(enabled) is bool, 'Explicit Boolean enabled flag required')
        self.parent = parent
        self.enabled = enabled
        if enabled:
            require(residual is not None and hasattr(residual, 'manifest') and (alignment is not None) and isinstance(parent_spec, BoundSpec) and isinstance(original_spec, BoundSpec) and (original_head is not None), 'Bound enabled residual inputs required')
            require(hashlib.sha256(Path(alignment.__file__).read_bytes()).hexdigest() == ALIGNMENT_SHA, 'Alignment source differs')
            self.residual = residual
            self.alignment = alignment
            self.spec = deepcopy(parent_spec.value)
            self.v9spec = deepcopy(original_spec.value)
            self.original_head = original_head
            require(parent_spec.sha256 == residual.manifest['parent']['spec_sha256'] and original_spec.sha256 == residual.manifest['parent']['original_v9_spec_sha256'], 'Parent/v9 spec identity differs')
            require(self.spec['language'] == self.v9spec['language'] == residual.manifest['language'], 'Parent/residual language differs')
            self.spec_identity = parent_spec.canonical
            self.v9_spec_identity = original_spec.canonical
            require(digest(self.spec) == self.spec_identity and digest(self.v9spec) == self.v9_spec_identity, 'Bound spec object mutated before attachment')

    def scores(self, prefixes, spec):
        if not self.enabled:
            return self.parent.scores(prefixes, spec)
        raise ValueError('Enabled residual uses predict_many: original-v9 fallback must keep its own threshold and identity')

    def predict_many(self, payloads, *, focuses=None):
        from roguard.gemma4_review import _LOCK
        with _LOCK:
            return self._predict_many(payloads, focuses=focuses)

    def _predict_many(self, payloads, *, focuses=None):
        require(self.enabled, 'Disabled engine delegates scores; use the configured parent public model for predictions')
        require(isinstance(payloads, list) and 1 <= len(payloads) <= 128, 'One to128 payloads required')
        if focuses is None:
            focuses = [None] * len(payloads)
        require(isinstance(focuses, list) and len(focuses) == len(payloads), 'Focus inventory differs')
        require(digest(self.spec) == self.spec_identity and digest(self.v9spec) == self.v9_spec_identity, 'Bound policy spec changed')
        import numpy as np
        from roguard.gemma4_contract import PREFIX_TOKENS, classifier_tokens
        from roguard.gemma4_attention import capture_with_prefix, replay
        from roguard.gemma4_review import _score
        from roguard.interpreter import prepare_input
        mx = self.parent.mx
        model = self.parent.model
        tokenizer = self.parent.tokenizer
        hf = getattr(tokenizer, '_tokenizer', tokenizer)
        require(callable(hf) and hf.is_fast, 'Offset tokenizer must be callable fast tokenizer')
        records = []
        for p, focus in zip(payloads, focuses):
            prepared, selected_focus, input_identity = prepare_input(p, focus)
            require(prepared['language'] == self.spec['language'], 'Language differs')
            sequence = classifier_tokens(tokenizer, prepared)
            require(sequence[:len(PREFIX_TOKENS)] == PREFIX_TOKENS and len(sequence) > len(PREFIX_TOKENS), 'Fixed classifier prefix differs')
            route = route_for_length(len(sequence), self.residual.manifest['maximum_tokens'])
            a = None
            if route == 'residual':
                a = self.alignment.align(prepared, hf)
                require(a['classifier_token_ids'] == sequence and a['classifier_ids_unchanged'] is True and (a['token_truncation_applied'] is False), 'Alignment token contract differs')
            records.append((route, sequence, a, selected_focus, input_identity))
        cache = None
        output = []
        state = feature = logits = value = key = None
        try:
            if any((route != 'original_v9_unsupported' for route, *_ in records)):
                cache = self.parent.make_cache(model.base)
                mx.eval(model.base.model(mx.array([PREFIX_TOKENS]), cache=cache))
            for route, sequence, a, focus, identity in records:
                spec = self.spec if route == 'residual' else self.v9spec
                model_identity = self.residual.identity if route == 'residual' else self.residual.manifest['parent']['original_v9_spec_sha256']
                format_name = 'roguard-role-residual-review-v1' if route == 'residual' else 'roguard-gemma4-tail-review-v1'
                if route == 'original_v9_unsupported':
                    score = None
                else:
                    state = capture_with_prefix(model.base, mx.array([sequence[len(PREFIX_TOKENS):]]), cache, start=24)
                    mx.eval(state)
                    feature = replay(model.base, state)[:, 0, :]
                    head = model.heads[self.spec['language']] if route == 'residual' else self.original_head
                    logits = head(feature)
                    mx.eval(feature, logits)
                    if route == 'residual':
                        require(state['start'] == 24 and state['layers'] == 42 and (state['length'] == len(sequence)) and (set(state['shared']) == {22, 23}), 'Shared capture architecture differs')
                        require(tuple(state['previous_kvs']) == tuple((23 if (i + 1) % 6 == 0 else 22 for i in range(24, 42))), 'Shared VALUE source differs')
                        key, value = state['shared'][23]
                        require(key.shape == value.shape == (1, 2, len(sequence), 512), 'Global VALUE length differs')
                        roles = value_pools(value, a['role_pool_positions'], mx, np)
                        meta = np.asarray([len(p) / 1024 for p in a['role_pool_positions']] + list(map(float, a['shared_structural_role_presence'])), dtype=np.float32)
                        logits = self.residual(np.asarray(logits).copy(), np.asarray(feature).copy(), roles[None, :, :], meta[None, :])
                    score = _score(logits[0], spec)
                output.append(render_result(score, spec, identity, model_identity, focus, format_name))
                state = feature = logits = value = key = None
                mx.clear_cache()
        finally:
            active_error = sys.exc_info()[1]
            state = feature = logits = value = key = cache = None
            try:
                mx.clear_cache()
            except Exception as cleanup_error:
                if active_error is None:
                    raise
                if hasattr(active_error, 'add_note'):
                    active_error.add_note('Cache cleanup also failed: ' + repr(cleanup_error))
        return output
PARENT75 = {'ro': '1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3', 'uk': '466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}
PARENT75_SPEC = {'ro': '5fdeae97843d873755ed41a02453b5c29823fd8028508c3e99d1586175c8f4a9', 'uk': 'bbf3fea8ee298c40d86701851fe045352ea1a924ba5287dd52b70e59ce25de27'}
V9 = {'ro': '114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09', 'uk': '0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f'}
V9_SPEC = {'ro': '6b2ea2979ffa43c735d8f8a6e3f7e1fbf6cf5c085889e8d484a13bf43026ccc5', 'uk': 'fc86fed9981b786149c4f6855c3d98f07067654cf2af05d40aef589345be34ed'}

def parameter_proof(model, language, np, mx, flatten):
    excluded = 'heads.' + language + '.'
    objects = {k: [id(v), list(v.shape), str(v.dtype)] for k, v in flatten(model.parameters()) if not k.startswith(excluded)}
    h = hashlib.sha256()
    for k, v in sorted(flatten(model.trainable_parameters())):
        if k.startswith(excluded):
            continue
        mx.eval(v)
        a = np.asarray(v)
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return dict(other_parameter_identity_sha256=digest(objects), other_trainable_value_sha256=h.hexdigest(), frozen_base_values_hashed=False)

def install75_and_retain_v9(model, language, tensors, np, mx, flatten):
    """Replace only existing four75 head tensors, retain an independent v9 head."""
    existing = dict(flatten(model.heads[language].parameters()))
    require(set(existing) == set(SHAPES), 'Existing parent head inventory differs')
    original_head = deepcopy(model.heads[language])
    original_values = {k: np.asarray(v).copy() for k, v in existing.items()}
    original_copied = dict(flatten(original_head.parameters()))
    require(set(original_copied) == set(existing) and all((np.asarray(original_copied[k]).tobytes() == v.tobytes() for k, v in original_values.items())), 'Original-v9 head snapshot differs')
    values = {k: tensors['heads.' + language + '.' + k] for k in existing}
    for k, v in values.items():
        require(v.dtype == np.float32 and v.shape == existing[k].shape and np.isfinite(v).all(), 'Invalid75 head tensor')
    before = parameter_proof(model, language, np, mx, flatten)
    model.heads[language].load_weights([(k, mx.array(v)) for k, v in values.items()], strict=True)
    model.eval()
    mx.eval(model.heads[language].parameters())
    after = parameter_proof(model, language, np, mx, flatten)
    require(before == after, '75 head installation changed other parameters')
    installed = dict(flatten(model.heads[language].parameters()))
    require(all((np.asarray(installed[k]).dtype == np.float32 and np.asarray(installed[k]).tobytes() == v.tobytes() for k, v in values.items())), 'Installed75 head bytes differ')
    require(all((np.asarray(dict(flatten(original_head.parameters()))[k]).tobytes() == v.tobytes() for k, v in original_values.items())), 'Retained original-v9 head changed during75 install')
    return (original_head, after)

def _from_local_files(*, base, original_v9_adapter, parent75_weights, parent75_spec, original_v9_spec, residual_manifest, residual_tensors, expected_residual_manifest_sha256):
    """Explicit local experimental factory. This function loads ONE neural backbone.
 Not invoked by source/synthetic tests; native parity remains required separately.
 """
    import numpy as np
    from safetensors.numpy import load
    from mlx.utils import tree_flatten
    from roguard.gemma4_review import Gemma4ReviewModel, _Engine
    residual = ResidualArrays.from_files(residual_manifest, residual_tensors, expected_manifest_sha256=expected_residual_manifest_sha256)
    manifest = residual.manifest
    language = manifest['language']
    parent = manifest['parent']
    require(parent['original_v9_adapter_sha256'] == V9[language] and parent['original_v9_spec_sha256'] == V9_SPEC[language] and (parent['head75_source_sha256'] == PARENT75[language]) and (parent['spec_sha256'] == PARENT75_SPEC[language]), 'Only frozen original-v9 backbone/spec and unchanged75 parent allowed')
    spec75 = BoundSpec(parent75_spec, PARENT75_SPEC[language])
    spec9 = BoundSpec(original_v9_spec, V9_SPEC[language])
    require(spec75.value['mode'] == 'coupled_policy', 'Frozen75 requires coupled policy')
    checked75 = Gemma4ReviewModel(parent75_spec, parent75_weights, base, expected_sha256=PARENT75_SPEC[language])
    checked9 = Gemma4ReviewModel(original_v9_spec, original_v9_adapter, base, expected_sha256=V9_SPEC[language])
    require(checked75._engine is None and checked9._engine is None, 'Parent validation unexpectedly loaded engine')
    raw75 = bounded(parent75_weights, 16000000)
    require(hashlib.sha256(raw75).hexdigest() == PARENT75[language], '75 source bytes changed')
    tensors75 = load(raw75)
    raw9 = bounded(original_v9_adapter, 16000000)
    require(hashlib.sha256(raw9).hexdigest() == V9[language], 'Original-v9 adapter bytes changed')
    del raw9
    from . import role_alignment as alignment
    require(hashlib.sha256(Path(alignment.__file__).read_bytes()).hexdigest() == ALIGNMENT_SHA, 'Packaged alignment source bytes differ')
    engine = _Engine(base, original_v9_adapter, V9[language])
    try:
        original_head, proof = install75_and_retain_v9(engine.model, language, tensors75, np, engine.mx, tree_flatten)
        result = OptionalResidualEngine(engine, enabled=True, residual=residual, alignment=alignment, parent_spec=spec75, original_spec=spec9, original_head=original_head)
        result.installation_proof = proof
        return result
    except BaseException:
        del engine
        import gc
        gc.collect()
        raise

def _load_engine(**kwargs):
    from roguard.gemma4_review import _LOCK
    with _LOCK:
        return _from_local_files(**kwargs)

class ExperimentalRoleResidualModel:
    """Explicit local experiment, never selected by the normal model profile.

 Disabled mode delegates the configured public parent model unchanged. Enabled
 mode uses the613-token residual route and original-v9 fallback through1024.
 """

    def __init__(self, parent, *, enabled):
        require(type(enabled) is bool, 'Explicit Boolean enabled flag required')
        require(parent is not None and callable(getattr(parent, 'predict_many', None)), 'A configured parent/model is required')
        self._parent = parent
        self.enabled = enabled

    def predict(self, payload, *, focus=None):
        if not self.enabled:
            return self._parent.predict(payload, focus=focus)
        return self._parent.predict_many([payload], focuses=[focus])[0]

    def predict_many(self, payloads, *, focuses=None):
        return self._parent.predict_many(payloads, focuses=focuses)

    @property
    def profile(self):
        return dict(format='roguard-experimental-role-residual-profile-1', experimental=True, qualified=False, enabled=self.enabled, residual_maximum_tokens=613 if self.enabled else None, original_v9_maximum_tokens=1024 if self.enabled else None, above_residual_envelope='original_v9' if self.enabled else 'configured_parent', fallback_on_error=False, fallback_on_low_confidence=False, default_profile_changed=False)

def load_experimental_role_residual(*, enabled, parent=None, **assets):
    """Load only explicitly supplied local experiment assets, or delegate a parent.

 No downloads or weight discovery occur. Enabling requires the bound residual,
 frozen75 parent and originalv9 assets described in docs/EXPERIMENTAL_RESIDUAL.md.
 Disabled mode does not inspect assets or import optional numerical dependencies.
 """
    require(type(enabled) is bool, 'Explicit Boolean enabled flag required')
    if not enabled:
        return ExperimentalRoleResidualModel(parent, enabled=False)
    require(parent is None, 'Enabled mode constructs its pinned one-backbone parent')
    return ExperimentalRoleResidualModel(_load_engine(**assets), enabled=True)
