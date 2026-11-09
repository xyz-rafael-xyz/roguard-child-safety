"""Explicit local role-reader experiment; no qualified model is distributed.

The parent stays original v9. A frozen token reader supplies five log-odds
and an availability bit to a small correction of all nineteen parent logits.
Both numerical cores are byte-identical to the registered CPU inference.
"""
from collections import namedtuple
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import sys

FORMAT = 'roguard-canonical-reader-residual-experiment-1'
ReaderPrediction = namedtuple('ReaderPrediction', 'response logits reader_features')
V9 = {
    'ro': '114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
    'uk': '0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f',
}
V9_SPEC = {
    'ro': '6b2ea2979ffa43c735d8f8a6e3f7e1fbf6cf5c085889e8d484a13bf43026ccc5',
    'uk': 'fc86fed9981b786149c4f6855c3d98f07067654cf2af05d40aef589345be34ed',
}
CORE_HASHES = {
    '_canonical_role_reader.py': '105528da80a521be62d339e4e5e6ca6d522b1f72b8c9ff04366539bb2bb3741e',
    '_canonical_reader_residual.py': '0b44ae527b8294e87b454430e55a7bda89857fd411dc6adf7ec37d818d55dd17',
    'role_alignment.py': '0c65cfcb800a445c52e9c880475e485ea9d35e8241468c92b7f861c37658ad6a',
}
ARRAY_ROLES = ('reader_weights', 'reader_stats', 'residual_weights', 'residual_stats')
POOLS = ('current_minor', 'prior_minor', 'prior_nonminor')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _bytes(path, expected, limit):
    require(_hash(expected), 'Expected SHA256 required')
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit and hashlib.sha256(raw).hexdigest() == expected,
            'Bound asset bytes differ: ' + str(path))
    return raw


def _arrays(path, expected):
    import numpy as np
    with np.load(io.BytesIO(_bytes(path, expected, 4 * 1024**2)), allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def validate_manifest(value):
    keys = {'format', 'language', 'qualified', 'parent_adapter_sha256',
            'parent_spec_sha256', 'arrays', 'fit_report_sha256', 'reader_report_sha256'}
    require(isinstance(value, dict) and set(value) == keys, 'Exact reader manifest required')
    language = value['language']
    require(value['format'] == FORMAT and isinstance(language, str)
            and language in V9 and value['qualified'] is False,
            'Only explicit unqualified reader experiments are supported')
    require(value['parent_adapter_sha256'] == V9[language]
            and value['parent_spec_sha256'] == V9_SPEC[language], 'Original v9 parent required')
    require(isinstance(value['arrays'], dict) and set(value['arrays']) == set(ARRAY_ROLES)
            and all(_hash(h) for h in value['arrays'].values()), 'Exact four learned assets required')
    require(_hash(value['fit_report_sha256']) and _hash(value['reader_report_sha256']),
            'Completed fitting and reader provenance required')
    return deepcopy(value)


def role_slots(prefix, alignment):
    """Map actual content assignments to reader slots; no authored labels enter."""
    turns = prefix['turns']
    focus = len(turns) - 1
    require(turns[focus]['role'] == 'minor' and tuple(alignment['role_pool_names']) == POOLS,
            'Actual minor focus and role pool contract required')
    positions = alignment['eligible_content_positions']
    assignments = alignment['assignments']
    require(positions == sorted(set(positions))
            and [a['token_index'] for a in assignments] == positions,
            'Ordered eligible token assignments required')
    slots = [[], [], []]
    for i, assignment in enumerate(assignments):
        turn = assignment['turn_index']
        require(type(turn) is int and 0 <= turn <= focus
                and assignment['role'] == turns[turn]['role'], 'Assignment turn and role differ')
        pool = ('current_minor' if turn == focus else
                'prior_minor' if assignment['role'] == 'minor' else 'prior_nonminor')
        require(assignment['pool'] == pool, 'Assignment pool differs')
        slots[POOLS.index(pool)].append(i)
    require([[positions[i] for i in slot] for slot in slots] == alignment['role_pool_positions'],
            'Actual role pool coordinates differ')
    return positions, slots


def project_features(reader, q, values, prefix, alignment):
    """Same host VALUE layout and canonical reader as the training stream."""
    import numpy as np
    positions, slots = role_slots(prefix, alignment)
    count = alignment['classifier_token_count']
    require(values.dtype == np.float32 and values.shape == (2, count, 512)
            and np.isfinite(values).all(), 'Complete finite F32 global VALUE required')
    require(all(type(i) is int and 387 <= i < count for i in positions),
            'Eligible positions outside classifier content')
    packed = values[:, positions, :].transpose(1, 0, 2).reshape(len(positions), 1024).copy()
    return reader.project(q, packed, slots)


class ReaderResidualModel:
    """Lazy one-backbone model with explicitly supplied, hash-bound local assets.

    Loading this class does not constitute release qualification. Prediction
    uses the original v9 policy and threshold, including its uncalibrated score.
    """

    def __init__(self, *, manifest, expected_manifest_sha256, base, parent_adapter,
                 parent_spec, reader_weights, reader_stats, residual_weights, residual_stats):
        value = validate_manifest(json.loads(_bytes(manifest, expected_manifest_sha256, 128 * 1024)))
        for name, expected in CORE_HASHES.items():
            _bytes(Path(__file__).with_name(name), expected, 128 * 1024)
        from ._canonical_role_reader import Reader
        from ._canonical_reader_residual import Residual
        from .gemma4_review import Gemma4ReviewModel
        paths = dict(reader_weights=reader_weights, reader_stats=reader_stats,
                     residual_weights=residual_weights, residual_stats=residual_stats)
        arrays = {name: _arrays(paths[name], value['arrays'][name]) for name in ARRAY_ROLES}
        self._reader = Reader(arrays['reader_weights'], arrays['reader_stats'])
        self._residual = Residual(arrays['residual_weights'], arrays['residual_stats'])
        self._parent = Gemma4ReviewModel(parent_spec, parent_adapter, base,
                                       expected_sha256=value['parent_spec_sha256'])
        require(self._parent.language == value['language']
                and self._parent._spec['adapter_sha256'] == value['parent_adapter_sha256'],
                'Parent language and adapter identity differ')
        self.identity = expected_manifest_sha256
        self.language = value['language']
        self._manifest = value

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def predict(self, payload, *, focus=None):
        return self.predict_many([payload], focuses=[focus])[0]

    def predict_many(self, payloads, *, focuses=None):
        return [detail.response for detail in self.predict_details(payloads, focuses=focuses)]

    def predict_details(self, payloads, *, focuses=None):
        """Same-call immutable F32 snapshots for explicit numerical auditing."""
        from .gemma4_review import _LOCK
        with _LOCK:
            return self._predict_details(payloads, focuses)

    def _predict_details(self, payloads, focuses):
        import numpy as np
        from .gemma4_review import _Engine, _score
        from .gemma4_contract import PREFIX_TOKENS, classifier_tokens
        from .gemma4_attention import capture_with_prefix, replay
        from .interpreter import prepare_input
        from .role_alignment import align
        from .experimental_role_residual import render_result
        require(isinstance(payloads, list) and 1 <= len(payloads) <= 128,
                'One to128 conversations required')
        if focuses is None:
            focuses = [None] * len(payloads)
        require(isinstance(focuses, list) and len(focuses) == len(payloads), 'Focal indices differ')
        prepared = [prepare_input(p, f) for p, f in zip(payloads, focuses)]
        require(all(p[0]['language'] == self.language for p in prepared), 'Model language differs')
        if self._parent._engine is None:
            self._parent._engine = _Engine(*self._parent._args)
        engine = self._parent._engine
        mx, model = engine.mx, engine.model
        tokenizer = getattr(engine.tokenizer, '_tokenizer', engine.tokenizer)
        require(callable(tokenizer) and tokenizer.is_fast, 'Fast offset tokenizer required')
        records = []
        for prefix, focus, identity in prepared:
            sequence = classifier_tokens(engine.tokenizer, prefix)
            require(sequence[:len(PREFIX_TOKENS)] == PREFIX_TOKENS
                    and len(sequence) > len(PREFIX_TOKENS), 'Fixed classifier prefix differs')
            alignment = None
            if len(sequence) <= 1024:
                alignment = align(prefix, tokenizer)
                require(alignment['classifier_token_ids'] == sequence
                        and alignment['classifier_ids_unchanged'] is True
                        and alignment['token_truncation_applied'] is False,
                        'Actual alignment differs from classifier input')
            records.append((prefix, focus, identity, sequence, alignment))
        cache = state = q = parent = key = value = converted = None
        results = []
        try:
            if any(r[-1] is not None for r in records):
                cache = engine.make_cache(model.base)
                mx.eval(model.base.model(mx.array([PREFIX_TOKENS]), cache=cache))
            for prefix, focus, identity, sequence, alignment in records:
                raw = features = None
                if alignment is not None:
                    state = capture_with_prefix(model.base, mx.array([sequence[len(PREFIX_TOKENS):]]),
                                                deepcopy(cache), start=24)
                    mx.eval(state)
                    q = replay(model.base, state)[:, 0, :]
                    parent = model.heads[self.language](q)
                    mx.eval(q, parent)
                    require(state['start'] == 24 and state['layers'] == 42
                            and state['length'] == len(sequence) and set(state['shared']) == {22, 23}
                            and tuple(state['previous_kvs']) == tuple(
                                23 if (i+1) % 6 == 0 else 22 for i in range(24, 42)),
                            'Original shared VALUE architecture differs')
                    key, value = state['shared'][23]
                    require(key.shape == value.shape == (1, 2, len(sequence), 512),
                            'Complete global VALUE axis differs')
                    converted = value.astype(mx.float32)
                    mx.eval(converted)
                    host_q = np.asarray(q)[0].copy()
                    features = project_features(self._reader, host_q, np.asarray(converted)[0].copy(),
                                                prefix, alignment)
                    raw = self._residual.predict(host_q, features, np.asarray(parent)[0].copy())
                score = None if raw is None else _score(raw, self._parent._spec)
                response = render_result(score, self._parent._spec, identity, self.identity, focus,
                                         'roguard-canonical-reader-review-v1')
                results.append(ReaderPrediction(response, None if raw is None else tuple(map(float, raw)),
                                               None if features is None else tuple(map(float, features))))
                state = q = parent = key = value = converted = None
                mx.clear_cache()
        finally:
            active_error = sys.exc_info()[1]
            cache = state = q = parent = key = value = converted = None
            try:
                mx.clear_cache()
            except Exception as cleanup_error:
                if active_error is None:
                    raise
                if hasattr(active_error, 'add_note'):
                    active_error.add_note('Native cache cleanup also failed: ' + repr(cleanup_error))
        return results

    def close(self):
        """Release the optional native model; a later prediction loads it again."""
        from .gemma4_review import _LOCK
        with _LOCK:
            if self._parent._engine is not None:
                engine = self._parent._engine
                self._parent._engine = None
                engine.model = None
                engine.mx.clear_cache()
