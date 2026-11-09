"""Explicit offline v10 experimental bundle. No downloads or promotion flags.

Only this loader's active instance is leased; unrelated low-level model users are
outside that guarantee. Use the context manager or close() before another loader.
Calibration belongs to outer configuration, not the inference pipeline identity.
"""
from copy import deepcopy
from pathlib import Path, PurePosixPath
import gc
import hashlib
import json
import shutil
import tempfile
import threading
import traceback
import unicodedata

FORMAT = 'roguard-public-v10-bundle-1'
NATIVE_FORMAT = 'roguard-public-v10-bundle-2'
_LEASE_LOCK = threading.RLock()
_OWNER = None
_POISONED = False


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _valid_sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _read(path, maximum):
    with Path(path).open('rb') as source:
        raw = source.read(maximum + 1)
    _require(len(raw) <= maximum, 'Bundle metadata exceeds bound')
    return raw


def _relative(root, binding, maximum):
    _require(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}, 'Exact relative asset binding required')
    name = binding['path']
    _require(type(name) is str and name and unicodedata.normalize('NFC', name) == name and ':' not in name and '\\' not in name and '\x00' not in name, 'Portable relative asset path required')
    rel = PurePosixPath(name)
    _require(not rel.is_absolute() and all(p not in ('', '.', '..') for p in name.split('/')), 'Unsafe relative asset path')
    _require(type(binding['bytes']) is int and 0 < binding['bytes'] <= maximum and _valid_sha(binding['sha256']), 'Bound asset size/hash required')
    target = root
    for part in rel.parts:
        target = target / part
        _require(not target.is_symlink(), 'Bundle metadata symlinks are not accepted')
    _require(target.is_file() and target.resolve().is_relative_to(root) and target.stat().st_size == binding['bytes'], 'Missing or wrong-sized bundle asset')
    _require(_sha(target) == binding['sha256'], 'Bundle asset digest differs')
    return target


def _schema(spec):
    _require(type(spec) is dict and set(spec) == {'format', 'qualified', 'languages', 'inventories', 'structured', 'response', 'calibration'}, 'Exact v10 bundle schema required')
    _require(spec['format'] in (FORMAT, NATIVE_FORMAT) and spec['qualified'] is False and spec['languages'] == ['ro', 'uk'], 'Explicit experimental bilingual bundle required')
    _require(type(spec['inventories']) is dict and set(spec['inventories']) == {'structured', 'response'}, 'Both approved runtime inventories required')
    s = spec['structured']; r = spec['response']
    structured_keys = {'mode', 'budget', 'pipeline_digest'}
    if spec['format'] == NATIVE_FORMAT:
        structured_keys.add('backend')
        _require(type(s) is dict and s.get('backend') == 'native_tail19', 'Explicit native-tail backend required')
    _require(type(s) is dict and set(s) == structured_keys and s['mode'] in ('direct-first', 'uniform') and type(s['budget']) is int and s['budget'] == 1200 and _valid_sha(s['pipeline_digest']), 'Structured mode/identity contract differs')
    _require(type(r) is dict and set(r) == {'reference_identity', 'parent_identity', 'artifact_identity'} and all(_valid_sha(v) for v in r.values()), 'Response identities required')


def _calibration(value, pipeline, language):
    if value is None:
        return None
    _require(type(value) is dict, 'Calibration must be an explicit artifact object')
    from roguard.trajectory_sets import predict_set
    copied = deepcopy(value)
    predict_set([1 / 432] * 432, copied, language=language, pipeline_digest=pipeline, supported=False)
    return copied


def _request(payload, focus):
    from roguard.trajectory_data import validate_conversation, prefix
    validate_conversation(payload)
    focus = len(payload['turns']) - 1 if focus is None else focus
    _require(type(focus) is int, 'Explicit integer focus required')
    selected = prefix(payload, focus)
    _require(selected['turns'][-1]['role'] == 'minor', 'Focal minor required')
    return deepcopy(payload), focus


def _native_lock():
    from roguard.gemma4_review import _LOCK
    return _LOCK


class _Backend:
    """Own every native reference; detach it even if closing verification fails."""
    def __init__(self, closure):
        self.closure = closure
        self.wrapper = self.component = self.optional = self.engine = None
        self.offset = self.parent = self.reference = None
        self.closed = False

    def predict_many(self, requests):
        return self.wrapper.predict_many(requests)

    def review_response(self, draft, language, context):
        return self.offset.review_response(draft, language, context)

    def close(self):
        if self.closed:
            return
        self.closed = True
        failures = []
        mx = None
        try:
            self.closure.verify()
        except BaseException as exc:
            failures.append(exc)
        try:
            if self.wrapper is not None:
                self.wrapper.components.clear()
            if self.component is not None:
                from roguard.native_tail import NativeTail
                if type(self.component) is NativeTail:
                    self.component.close()
                else:
                    self.component._parent = None
            if self.optional is not None:
                self.optional.parent = self.optional.original_head = self.optional.residual = self.optional.alignment = None
            if self.offset is not None:
                self.offset.parent = None
            if self.parent is not None:
                self.parent._readout = None
                self.parent.reference = None
            if self.reference is not None:
                if self.engine is None:
                    self.engine = self.reference._engine
                self.reference._engine = None
            if self.engine is not None:
                mx = self.engine.mx
                self.engine.model = self.engine.tokenizer = self.engine.make_cache = None
        except BaseException as exc:
            failures.append(exc)
        finally:
            self.wrapper = self.component = self.optional = self.engine = None
            self.offset = self.parent = self.reference = None
            gc.collect()
            if mx is not None:
                try:
                    mx.clear_cache()
                except BaseException as exc:
                    failures.append(exc)
        if failures:
            raise BaseExceptionGroup('Bundle backend cleanup failed', failures)


class _ActualFactory:
    """Public entrypoint always uses these actual reviewed constructors."""
    def prepare(self, root, spec, directory):
        from roguard.v10_closure import rebase_closure
        from roguard.experimental_structured import Closure as StructuredClosure
        from roguard.response_offset.closure import Closure as ResponseClosure
        result = {}
        native = spec['format'] == NATIVE_FORMAT
        structured_cls = StructuredClosure
        if native:
            from roguard.v10_closure import NativeTailClosure
            structured_cls = NativeTailClosure
        for kind, cls in (('structured', structured_cls), ('response', ResponseClosure)):
            binding = spec['inventories'][kind]
            approved = _relative(root, binding, 32 << 20)
            bound = rebase_closure(approved, expected_sha256=binding['sha256'], bundle_root=root, output_path=directory / (kind + '.json'))
            expected_kind = 'native_tail' if native and kind == 'structured' else kind
            _require(bound.kind == expected_kind, 'Approved inventory task differs')
            closure = cls(bound.path, bound.sha256)
            _require(closure.identity == bound.receipt_sha256, 'Rebased runtime receipt differs')
            result[kind] = closure
        return result

    def activate(self, task, language, spec, closures):
        from roguard.gemma4_contract import FILES
        for closure in closures.values():
            closure.verify()
        closure = closures[task]
        paths = [closure.asset_path('base:' + name) for name in FILES]
        base = paths[0].parent
        _require(all(path == base / name for path, name in zip(paths, FILES)), 'Pinned base files must share one local directory')
        _require(shutil.disk_usage(tempfile.gettempdir()).free >= 8 * 1024 ** 3 + sum(v['bytes'] for v in FILES.values()), 'Insufficient native temporary-copy reserve')
        backend = _Backend(closure)
        try:
            if task == 'structured' and spec['format'] == NATIVE_FORMAT:
                from roguard.native_tail import load
                from roguard.native_tail_structured import NativeTailStructuredWrapper
                from roguard.trajectory_model import TrajectoryModel
                language_closure = closure.language_view(language)
                backend.component = load(enabled=True, closure=language_closure,
                    expected_closure_identity=language_closure.identity)
                selector = TrajectoryModel(closure.asset_path('selector'), expected_sha256=closure.asset_sha('selector'))
                backend.wrapper = NativeTailStructuredWrapper({language: backend.component}, selector, closure,
                    active_language=language, mode=spec['structured']['mode'], budget=spec['structured']['budget'])
                _require(backend.wrapper.identity == spec['structured']['pipeline_digest'], 'Actual native structured pipeline differs')
            elif task == 'structured':
                from roguard.experimental_role_residual import load_experimental_role_residual
                from roguard.experimental_structured import StructuredWrapper
                from roguard.trajectory_model import TrajectoryModel
                backend.component = load_experimental_role_residual(enabled=True, base=base,
                    original_v9_adapter=closure.asset_path(language + '_adapter9'),
                    parent75_weights=closure.asset_path(language + '_weights75'),
                    parent75_spec=closure.asset_path(language + '_spec75'),
                    original_v9_spec=closure.asset_path(language + '_spec9'),
                    residual_manifest=closure.asset_path(language + '_residual_manifest'),
                    residual_tensors=closure.asset_path(language + '_residual_tensors'),
                    expected_residual_manifest_sha256=closure.asset_sha(language + '_residual_manifest'))
                backend.optional = backend.component._parent
                backend.engine = backend.optional.parent
                selector = TrajectoryModel(closure.asset_path('selector'), expected_sha256=closure.asset_sha('selector'))
                backend.wrapper = StructuredWrapper({language: backend.component}, selector, closure, active_language=language, mode=spec['structured']['mode'], budget=spec['structured']['budget'])
                _require(backend.wrapper.identity == spec['structured']['pipeline_digest'], 'Actual structured pipeline differs')
            else:
                from roguard.gemma4_review import Gemma4ReviewModel
                from roguard.gemma4_response_attention import Gemma4ResponseAttentionModel
                from roguard.response_offset.runtime import Frozen59TextOffset
                from roguard.response_offset.closure import Closure
                backend.reference = Gemma4ReviewModel(closure.asset_path('reference_spec'), closure.asset_path('reference_weights'), base, expected_sha256=closure.asset_sha('reference_spec'))
                backend.parent = Gemma4ResponseAttentionModel(closure.asset_path('response59_spec'), closure.asset_path('response59_weights'), backend.reference, expected_sha256=closure.asset_sha('response59_spec'))
                backend.offset = Frozen59TextOffset(backend.parent, closure, closure_type=Closure)
                ids = spec['response']
                _require(backend.reference.identity == ids['reference_identity'] and backend.parent.identity == ids['parent_identity'] and backend.offset.artifact_identity == ids['artifact_identity'], 'Actual response pipeline differs')
            return backend
        except BaseException as original:
            try:
                backend.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup('Bundle activation and cleanup failed', [original, cleanup])
            raise


class V10Bundle:
    """Lazy offline orchestrator. Construct through load_v10(), then close()."""
    def __init__(self, path, expected, *, _factory):
        self._closed = False
        self._active = self._backend = None
        self._factory = _factory
        self._directory = None
        self._path = Path(path).resolve(strict=True)
        _require(_valid_sha(expected), 'Expected bundle SHA256 required')
        raw = _read(self._path, 1 << 20)
        _require(hashlib.sha256(raw).hexdigest() == expected, 'Bundle digest differs')
        self._sha = expected; self._spec = _json(raw); _schema(self._spec)
        self._root = self._path.parent
        for binding in self._spec['inventories'].values():
            _relative(self._root, binding, 32 << 20)
        self._default_calibration = None
        if self._spec['calibration'] is not None:
            cp = _relative(self._root, self._spec['calibration'], 4 << 20)
            self._default_calibration = _json(_read(cp, 4 << 20))
            for language in ('ro', 'uk'):
                _calibration(self._default_calibration, self._spec['structured']['pipeline_digest'], language)
        self._default_calibration_digest = _digest(self._default_calibration)
        self._config_digest = _digest(self._spec)
        self.inference_identity = _digest({k: self._spec[k] for k in ('format', 'languages', 'structured', 'response')})
        self.bundle_sha256 = expected
        self._directory = tempfile.TemporaryDirectory(prefix='roguard-v10-closure-')
        try:
            self._closures = _factory.prepare(self._root, deepcopy(self._spec), Path(self._directory.name))
            _require(set(self._closures) == {'structured', 'response'}, 'Both prepared task closures required')
            self._guard()
        except BaseException:
            self._directory.cleanup(); self._directory = None; self._closed = True
            raise

    def _guard(self):
        _require(not self._closed, 'V10 bundle is closed')
        _require(_digest(self._spec) == self._config_digest and _sha(self._path) == self._sha, 'Bundle configuration changed')
        _require(_digest(self._default_calibration) == self._default_calibration_digest, 'Bound default calibration changed')
        for binding in self._spec['inventories'].values():
            _relative(self._root, binding, 32 << 20)
        if self._spec['calibration'] is not None:
            _relative(self._root, self._spec['calibration'], 4 << 20)

    @property
    def profile(self):
        return dict(format=self._spec['format'], qualified=False, inference_identity=self.inference_identity,
            bundle_sha256=self.bundle_sha256, structured_pipeline_digest=self._spec['structured']['pipeline_digest'],
            response_artifact_identity=self._spec['response']['artifact_identity'],
            mode=self._spec['structured']['mode'], automatic_downloads=False,
            supports_plan=False, release_evidence_verified=False)

    def _dispose(self):
        global _OWNER, _POISONED
        backend = self._backend
        self._backend = self._active = None
        try:
            if backend is not None:
                backend.close()
        except BaseException:
            _POISONED = True
            raise
        finally:
            if _OWNER is self:
                _OWNER = None

    def _lease_check(self):
        _require(not _POISONED, 'Previous native cleanup/activation failed; restart this worker process')
        owner = _OWNER
        _require(owner is None or owner is self, 'Close the other active v10 bundle before loading another encoder')

    def _activate(self, task, language):
        global _OWNER, _POISONED
        self._guard()
        _require(not _POISONED, 'Previous native cleanup/activation failed; restart this worker process')
        owner = _OWNER
        _require(owner is None or owner is self, 'Close the other active v10 bundle before loading another encoder')
        if self._active == (task, language):
            return self._backend
        self._dispose()
        _OWNER = self
        try:
            self._backend = self._factory.activate(task, language, deepcopy(self._spec), self._closures)
            self._active = (task, language)
            return self._backend
        except BaseException:
            _POISONED = True; _OWNER = None
            raise

    def _failed(self, original):
        global _POISONED
        _POISONED = True
        traceback.clear_frames(original.__traceback__)
        try:
            self._dispose()
        except BaseException as cleanup:
            raise BaseExceptionGroup('Inference and cleanup failed', [original, cleanup])

    def predict(self, payload, *, focus=None, calibration=None):
        return self.predict_many([payload], focuses=[focus], calibration=calibration)[0]

    def predict_many(self, payloads, *, focuses=None, calibration=None):
        _require(type(payloads) is list and payloads, 'Nonempty payload list required')
        if focuses is None:
            focuses = [None] * len(payloads)
        _require(type(focuses) is list and len(focuses) == len(payloads), 'Focus count differs')
        requests = []
        for payload, focus in zip(payloads, focuses):
            value, selected = _request(payload, focus)
            cal = _calibration(self._default_calibration if calibration is None else calibration, self._spec['structured']['pipeline_digest'], value['language'])
            requests.append(dict(payload=value, focus=selected, calibration=cal))
        # Always acquire native lock before the loader lease; no inverted order.
        with _native_lock(), _LEASE_LOCK:
            self._guard(); self._lease_check(); result = [None] * len(requests)
            try:
                for language in ('ro', 'uk'):
                    indices = [i for i, r in enumerate(requests) if r['payload']['language'] == language]
                    if not indices:
                        continue
                    backend = self._activate('structured', language)
                    values = backend.predict_many([requests[i] for i in indices])
                    _require(type(values) is list and len(values) == len(indices), 'Prediction count differs')
                    for i, value in zip(indices, values):
                        result[i] = value
                self._guard()
                return result
            except BaseException as original:
                self._failed(original)
                raise

    def review_response(self, draft, language, *, context):
        _require(language in ('ro', 'uk') and type(draft) is str and 1 <= len(draft.strip()) <= 6000 and len(draft) <= 6000 and type(context) is str and len(context) <= 12000, 'Explicit bounded draft/language/context required')
        draft.encode('utf-8'); context.encode('utf-8')
        with _native_lock(), _LEASE_LOCK:
            self._guard(); self._lease_check()
            try:
                backend = self._activate('response', language)
                result = backend.review_response(draft, language, context)
                self._guard()
                return result
            except BaseException as original:
                self._failed(original)
                raise

    def close(self):
        with _native_lock(), _LEASE_LOCK:
            if self._closed:
                return
            self._closed = True
            try:
                self._dispose()
            finally:
                if self._directory is not None:
                    self._directory.cleanup(); self._directory = None

    def __enter__(self):
        self._guard()
        return self

    def __exit__(self, kind, error, tb):
        try:
            self.close()
        except BaseException as cleanup:
            if error is not None:
                raise BaseExceptionGroup('Bundle operation and closing failed', [error, cleanup])
            raise
        return False


def load_v10(bundle, *, expected_sha256):
    """Verify local portable assets, without loading an encoder or downloading.

    The approved inventory must match this installed runtime. All low-level
    predictions retain qualified=False; this loader verifies no release proof.
    """
    return V10Bundle(bundle, expected_sha256, _factory=_ActualFactory())
