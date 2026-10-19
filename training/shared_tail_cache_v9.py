"""Non-executable, identity-bound storage for frozen shared-attention states."""
import json
from pathlib import Path
import re
import zipfile


def _identity(value):
    if (not isinstance(value, dict) or set(value) != {'token_sha256', 'source_freeze_sha256'}
            or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in value.values())):
        raise ValueError('Invalid cached input identity')
    return dict(value)


def _encode(state, identity):
    import numpy as np
    identity = _identity(identity)
    start, layers, length = state['start'], state['layers'], state['length']
    if any(type(v) is not int for v in (start, layers, length)) or not 0 < start < layers <= 128 or not 0 < length <= 4096:
        raise ValueError('Invalid cache dimensions')
    previous = tuple(state['previous_kvs'])
    tail = layers-start
    if len(previous) != tail or any(type(i) is not int or not 0 <= i < start for i in previous):
        raise ValueError('Invalid frozen KV sources')
    if len(state['per_layer']) != tail or len(state['masks']) != tail or set(state['shared']) != set(previous):
        raise ValueError('Incomplete tail cache')
    arrays = {'hidden': np.asarray(state['hidden'])}
    h = arrays['hidden']
    if h.ndim != 3 or h.shape[0:2] != (1, 1) or not h.shape[2]:
        raise ValueError('Cache must represent one final-token input')
    for index, pair in state['shared'].items():
        if len(pair) != 2:
            raise ValueError('Missing key/value state')
        key, value = map(np.asarray, pair)
        if key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1 or key.shape[2] != length or min(key.shape) <= 0:
            raise ValueError('Invalid complete key/value dimensions')
        arrays[f'key_{index}'], arrays[f'value_{index}'] = key, value
    per_layer, masks = [], []
    for i, value in enumerate(state['per_layer']):
        per_layer.append(value is not None)
        if value is not None:
            array = np.asarray(value)
            if array.ndim != 3 or array.shape[:2] != (1, 1) or not array.shape[2]:
                raise ValueError('Invalid per-layer input')
            arrays[f'input_{i}'] = array
    for i, value in enumerate(state['masks']):
        masks.append(value is not None)
        if value is not None:
            array = np.asarray(value)
            if array.dtype != np.bool_ or array.shape != (1, length):
                raise ValueError('Expected exact Boolean last-query mask')
            arrays[f'mask_{i}'] = array
    for name, array in arrays.items():
        if name.startswith('mask_'):
            continue
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Cache requires finite float32 states')
    metadata = dict(format='roguard-shared-tail-cache-v1', identity=identity, start=start, layers=layers,
        length=length, previous_kvs=list(previous), per_layer=per_layer, masks=masks)
    arrays['metadata'] = np.frombuffer(json.dumps(metadata, separators=(',', ':')).encode(), dtype=np.uint8)
    return arrays


def write_state(path, state, identity):
    import numpy as np
    arrays = _encode(state, identity)
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)


def read_state(path, *, expected_identity):
    import numpy as np
    expected_identity = _identity(expected_identity)
    path = Path(path)
    if not 0 < path.stat().st_size <= 128*1024**2:
        raise ValueError('Cache file exceeds size limit')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 400 or sum(v.file_size for v in entries) > 128*1024**2 or len({v.filename for v in entries}) != len(entries):
            raise ValueError('Invalid cache archive inventory')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    raw = arrays.pop('metadata', None)
    if raw is None or raw.dtype != np.uint8 or raw.ndim != 1 or not 0 < raw.size <= 8192:
        raise ValueError('Invalid cache metadata')
    metadata = json.loads(raw.tobytes())
    if (not isinstance(metadata, dict) or set(metadata) != {'format', 'identity', 'start', 'layers', 'length', 'previous_kvs', 'per_layer', 'masks'}
            or metadata['format'] != 'roguard-shared-tail-cache-v1' or _identity(metadata['identity']) != expected_identity):
        raise ValueError('Cache identity or format differs')
    if not isinstance(metadata['per_layer'], list) or not isinstance(metadata['masks'], list) or any(type(v) is not bool for v in metadata['per_layer']+metadata['masks']):
        raise ValueError('Invalid cache presence flags')
    previous = metadata['previous_kvs']
    if not isinstance(previous, list) or any(type(v) is not int for v in previous):
        raise ValueError('Invalid frozen KV references')
    state = {key: metadata[key] for key in ('start', 'layers', 'length')}
    try:
        state.update(previous_kvs=tuple(previous), hidden=arrays['hidden'],
            shared={i: (arrays[f'key_{i}'], arrays[f'value_{i}']) for i in set(previous)},
            per_layer=[arrays[f'input_{i}'] if present else None for i, present in enumerate(metadata['per_layer'])],
            masks=[arrays[f'mask_{i}'] if present else None for i, present in enumerate(metadata['masks'])])
    except KeyError as exc:
        raise ValueError('Incomplete cache arrays') from exc
    encoded = _encode(state, expected_identity)
    if set(encoded)-{'metadata'} != set(arrays):
        raise ValueError('Unexpected cache arrays')
    return state


def to_mlx(state):
    import mlx.core as mx
    return dict(state, hidden=mx.array(state['hidden']),
        shared={i: tuple(mx.array(v) for v in pair) for i, pair in state['shared'].items()},
        per_layer=[None if v is None else mx.array(v) for v in state['per_layer']],
        masks=[None if v is None else mx.array(v) for v in state['masks']])


def batch_states(states):
    """Batch equal-length cached inputs; never pad or change query positions."""
    import mlx.core as mx
    if not states or len(states) > 16:
        raise ValueError('Expected one to sixteen cached states')
    first = states[0]
    for state in states[1:]:
        if any(state[key] != first[key] for key in ('start', 'layers', 'length', 'previous_kvs')):
            raise ValueError('Batch requires identical positions and architecture')
        for left, right in zip(first['masks'], state['masks']):
            if (left is None) != (right is None) or left is not None and not bool(mx.array_equal(left, right).item()):
                raise ValueError('Batch attention masks differ')
        if [v is None for v in first['per_layer']] != [v is None for v in state['per_layer']]:
            raise ValueError('Batch per-layer inputs differ')
    return dict(first, hidden=mx.concatenate([s['hidden'] for s in states]),
        shared={i: tuple(mx.concatenate([s['shared'][i][j] for s in states]) for j in (0, 1)) for i in first['shared']},
        per_layer=[None if value is None else mx.concatenate([s['per_layer'][i] for s in states]) for i, value in enumerate(first['per_layer'])])
