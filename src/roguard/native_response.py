"""Explicit unqualified native74 response deployment; unchanged public model math.

The public bundle owns the single encoder lease. This loader verifies portable
assets before lazy construction, and never approves their quality or provenance.
"""
from collections import namedtuple
from copy import deepcopy
import hashlib
from pathlib import Path

from .assurance import digest
from .gemma4_contract import BASE_MODEL, BASE_REVISION
from .gemma4_response import LABELS, QUERY_SHA256
from .native_tail import (require, strict_json, valid_sha, tensor_schema as review_schema,
                          checked_tensors as review_tensors, SPEC9, WEIGHTS9)

FORMAT = 'roguard-native-response74-manifest-1'
PARENT_SPEC = 'a4a1afe83324c5f34ede2e4c8700fcd7213d9e48f4d607f1048e37ef059271c9'
PARENT_WEIGHTS = '9206896b6b9025865f0579a954b913a186380edd936239114ac178145474b848'
ASSETS = {'manifest', 'native_spec', 'native_adapter', 'reference_spec', 'reference_weights'}
PROVENANCE = {'selected_checkpoint_sha256', 'training_plan_sha256', 'training_report_sha256'}
DetailedResponse = namedtuple('DetailedResponse', 'response logits model_identity artifact_identity')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tensor_schema():
    return {**{k: v for k, v in review_schema().items() if k.startswith('base.')},
            'head.last.weight': [5, 2560], 'head.last.bias': [5]}


def source_digests():
    from . import gemma4_response_attention, gemma4_response, gemma4_review, gemma4_attention, gemma4_contract, native_tail
    modules = {'runtime': __file__, 'gemma4_response_attention': gemma4_response_attention.__file__,
               'gemma4_response': gemma4_response.__file__, 'gemma4_review': gemma4_review.__file__,
               'gemma4_attention': gemma4_attention.__file__, 'gemma4_contract': gemma4_contract.__file__,
               'native_tail': native_tail.__file__}
    return {k: sha(Path(path).read_bytes()) for k, path in modules.items()}


def checked_tensors(raw, inventory):
    import numpy as np
    from safetensors.numpy import load
    values = load(raw); schema = tensor_schema()
    require(type(inventory) is dict and set(values) == set(inventory) == set(schema), 'Exact74 tensor inventory')
    for name, value in values.items():
        require(value.dtype == np.float32 and list(value.shape) == schema[name] and np.isfinite(value).all(),
                'Finite float32 native74 shape')
        require(inventory[name] == dict(dtype='float32', shape=schema[name], sha256=sha(value.tobytes())),
                'Native74 tensor proof mismatch')
    return deepcopy(inventory)


def load(*, enabled, configured_parent=None, closure=None, expected_closure_identity=None):
    require(type(enabled) is bool, 'Explicit enabled boolean')
    if not enabled:
        require(configured_parent is not None, 'Disabled configured parent required')
        return configured_parent
    from .v10_closure import NativeResponseClosure
    require(type(closure) is NativeResponseClosure and valid_sha(expected_closure_identity),
            'Externally approved native response closure required')
    closure.verify(); require(closure.identity == expected_closure_identity, 'Closure identity mismatch')
    raw = {}
    for key in sorted(ASSETS):
        path = closure.asset_path(key); require(path.stat().st_size <= 16_000_000, 'Bounded response asset')
        raw[key] = path.read_bytes(); require(sha(raw[key]) == closure.asset_sha(key), 'Response asset changed')
    manifest = strict_json(raw['manifest'])
    keys = {'format', 'qualified', 'selected_arm', 'native_spec_sha256', 'adapter_sha256',
            'reference_spec_sha256', 'reference_adapter_sha256', 'original59_spec_sha256',
            'original59_adapter_sha256', 'tensor_inventory', 'runtime_sources', 'provenance'}
    require(type(manifest) is dict and set(manifest) == keys and manifest['format'] == FORMAT and
            manifest['qualified'] is False and manifest['selected_arm'] == 'native_attention_primary',
            'Explicit unqualified primary native74 manifest')
    require(manifest['original59_spec_sha256'] == PARENT_SPEC and manifest['original59_adapter_sha256'] == PARENT_WEIGHTS,
            'Exact original59 training parent declaration')
    require(type(manifest['provenance']) is dict and set(manifest['provenance']) == PROVENANCE and
            all(valid_sha(v) for v in manifest['provenance'].values()) and
            manifest['provenance']['selected_checkpoint_sha256'] == manifest['adapter_sha256'],
            'Exact selected checkpoint and declared training lineage')
    require(manifest['runtime_sources'] == source_digests(), 'Reviewed runtime source identity')
    for key, field in [('native_spec', 'native_spec_sha256'), ('native_adapter', 'adapter_sha256'),
                       ('reference_spec', 'reference_spec_sha256'), ('reference_weights', 'reference_adapter_sha256')]:
        require(sha(raw[key]) == manifest[field], 'Manifest asset digest edge')
    require(manifest['reference_spec_sha256'] == SPEC9['ro'] and manifest['reference_adapter_sha256'] == WEIGHTS9['ro'],
            'Exact original RO reference; never candidate main-task weights')
    from .gemma4_response_attention import FORMAT as RESPONSE_FORMAT, Gemma4ResponseAttentionModel
    spec = strict_json(raw['native_spec'])
    require(spec == dict(format=RESPONSE_FORMAT, base_model=BASE_MODEL, base_revision=BASE_REVISION,
                         query_sha256=QUERY_SHA256, adapter_sha256=manifest['adapter_sha256'],
                         boundary=24, hidden=0, labels=list(LABELS), qualified=False), 'Exact unchanged9-field runtime spec')
    proof = checked_tensors(raw['native_adapter'], manifest['tensor_inventory'])
    original = review_tensors(raw['reference_weights'])
    original_proof = {k: dict(dtype='float32', shape=list(v.shape), sha256=sha(v.tobytes())) for k, v in original.items()}
    del original, raw
    from .gemma4_review import Gemma4ReviewModel
    reference = Gemma4ReviewModel(closure.asset_path('reference_spec'), closure.asset_path('reference_weights'),
                                 closure.base_path(), expected_sha256=manifest['reference_spec_sha256'])
    parent = Gemma4ResponseAttentionModel(closure.asset_path('native_spec'), closure.asset_path('native_adapter'),
                                        reference, expected_sha256=manifest['native_spec_sha256'])
    closure.verify(); require(closure.identity == expected_closure_identity, 'Closing response closure identity')
    return NativeResponse(parent, closure, manifest, proof, original_proof)


class NativeResponse:
    """One unchanged parent public call; immutable same-call raw5 in detailed mode.

    The response dictionary remains the ordinary public dictionary, not a deeply
    immutable object. A root-approved manifest is still not a release certificate.
    """
    def __init__(self, parent, closure, manifest, proof, original_proof):
        self.parent, self.closure = parent, closure
        self.manifest, self._proof, self._original_proof = deepcopy(manifest), deepcopy(proof), deepcopy(original_proof)
        self.identity, self.artifact_identity = parent.identity, closure.asset_sha('manifest')
        self._closure_identity = closure.identity; self._state = digest(self.manifest)
        self._parent_state = (parent.identity, digest(parent._spec), str(parent._weights), parent._reference_binding)
        self._original_state = digest(self._original_proof)
        self._closed = False

    def _verify(self, full=True):
        from .gemma4_response_attention import Gemma4ResponseAttentionModel
        require(not self._closed and type(self.parent) is Gemma4ResponseAttentionModel and
                getattr(self.parent.review_response, '__func__', None) is Gemma4ResponseAttentionModel.review_response,
                'Actual response model implementation required')
        self.closure.verify() if full else self.closure.cheap_verify()
        require(self._proof == self.manifest['tensor_inventory'] and digest(self._original_proof) == self._original_state,
                'Retained tensor verification configuration changed')
        require(self.closure.identity == self._closure_identity and self.artifact_identity == self.closure.asset_sha('manifest') and
                digest(self.manifest) == self._state and self.manifest['runtime_sources'] == source_digests(),
                'Bound response configuration changed')
        require((self.parent.identity, digest(self.parent._spec), str(self.parent._weights), self.parent._reference_binding) ==
                self._parent_state and self.identity == self.parent.identity, 'Parent configuration changed')
        ref = self.parent.reference
        require((ref.identity, ref._args, digest(ref._spec)) == self.parent._reference_binding, 'Original reference changed')

    @staticmethod
    def _fingerprint(values, mx):
        import numpy as np
        mx.eval(values)
        result = {}
        for k, v in values.items():
            a = np.asarray(v); require(a.dtype == np.float32 and np.isfinite(a).all(), 'Finite native state')
            result[k] = dict(dtype='float32', shape=list(a.shape), sha256=sha(a.tobytes()))
        return result

    def _native_state(self):
        from mlx.utils import tree_flatten
        parent = self.parent; engine = parent.reference._engine; readout = parent._readout
        original = dict(tree_flatten(engine.model.trainable_parameters()))
        require(self._fingerprint(original, engine.mx) == self._original_proof, 'Exact original80 reference restoration')
        candidate = dict(readout.adapters, **{'head.last.weight': readout.head.weight, 'head.last.bias': readout.head.bias})
        require(self._fingerprint(candidate, engine.mx) == self._proof, 'Actual selected74 state changed')
        return {k: id(v) for k, v in original.items()}

    @staticmethod
    def _requests(rows):
        require(type(rows) is list and 1 <= len(rows) <= 96, 'Bounded one-to-96 response batch')
        for row in rows:
            require(type(row) is dict and set(row) == {'draft', 'language', 'context'}, 'Exact response request keys')
            draft, language, context = row['draft'], row['language'], row['context']
            require(language in ('ro', 'uk') and type(draft) is str and 1 <= len(draft.strip()) <= 6000 and
                    len(draft) <= 6000 and type(context) is str and len(context) <= 12000, 'Bounded response input')
            draft.encode('utf-8'); context.encode('utf-8')
        return deepcopy(rows)

    def review_response(self, draft, language, context=''):
        return self.review_response_detailed(draft, language, context).response

    def review_response_detailed(self, draft, language, context=''):
        return self.review_response_detailed_many([dict(draft=draft, language=language, context=context)])[0]

    def review_response_many(self, rows):
        return [value.response for value in self.review_response_detailed_many(rows)]

    def review_response_detailed_many(self, rows):
        """Sequential unchanged public forwards; full byte closure at batch boundaries.

        This amortizes verification, not encoder computation. Each row checks all
        original80 and selected74 tensors and exact restoration independently.
        A failed row or closing check invalidates the complete batch and wrapper.
        """
        from .gemma4_review import _LOCK
        rows = self._requests(rows)
        with _LOCK:
            self._verify()
            failure = None
            try:
                result = [self._one(row) for row in rows]
            except BaseException as error:
                failure = error
                raise
            finally:
                try:
                    self._verify()
                except BaseException as closing:
                    self._closed = True
                    if failure is not None:
                        raise BaseExceptionGroup('Response batch and closing verification failed', [failure, closing])
                    raise
                finally:
                    if failure is not None:
                        self._closed = True
            require(len(result) == len(rows), 'Response batch cardinality')
            return result

    def _one(self, row):
        from . import gemma4_response_attention as a
        import math
        self._verify(full=False); parent = self.parent; reference = parent.reference
        if reference._engine is None:
            reference._engine = a._Engine(*reference._args)
        if parent._readout is None:
            parent._readout = a._AdapterReadout(parent._weights, parent._spec['adapter_sha256'], reference._engine.model.base)
        require(type(parent._readout) is a._AdapterReadout, 'Reviewed readout required')
        saved_ids = self._native_state(); readout = parent._readout; captured = []
        class Observer:
            def __getattr__(self, name): return getattr(readout, name)
            def logits(self, features):
                values = readout.logits(features)
                require(type(values) is list and len(values) == 5 and all(type(v) is float and math.isfinite(v) for v in values),
                        'Finite same-call native5')
                captured.append(tuple(values)); return values
        failure = None
        try:
            parent._readout = Observer()
            response = parent.review_response(row['draft'], row['language'], row['context'])
        except BaseException as error:
            failure = error
            raise
        finally:
            parent._readout = readout
            readout = None
            try:
                require(self._native_state() == saved_ids, 'Original reference array aliases not restored')
                self._verify(full=False)
            except BaseException as closing:
                if failure is not None:
                    raise BaseExceptionGroup('Response call and closing verification failed', [failure, closing])
                raise
        require(len(captured) == (0 if response['abstained'] else 1) and response['model_identity'] == self.identity,
                'Exactly one public response head call when supported')
        return DetailedResponse(response, None if response['abstained'] else captured[0], self.identity, self.artifact_identity)
