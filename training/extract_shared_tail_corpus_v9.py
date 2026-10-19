"""Resumable frozen-body extraction for exact shared-KV tail adaptation."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import distribution, version
import json
from pathlib import Path
import time

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.gemma4_prompt_v9 import classifier_tokens
from training.gemma4_shared_tail_v9 import capture, replay
from training.gemma4_shared_prefix_v9 import capture_with_prefix
from training.shared_tail_cache_v9 import read_state, to_mlx, write_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    study = root/'shared-tail-corpus-164'
    freeze = digest(study/'source-freeze.json')
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen source or input differs: ' + path)
    base = json.loads((study/'base-manifest.json').read_text())
    if args.base.name != base['revision']:
        raise ValueError('Wrong base revision')
    for name, record in base['files'].items():
        if digest(args.base/name) != record['sha256']:
            raise ValueError('Pinned base differs')
    software = json.loads((study/'software.json').read_text())
    for name, expected in software['versions'].items():
        if version(name) != expected:
            raise ValueError('Dependency differs: ' + name)
    package = Path(distribution('mlx-lm').locate_file('mlx_lm'))
    for name, expected in software['implementation_sha256'].items():
        if digest(package/name) != expected:
            raise ValueError('Pinned implementation differs')
    import hashlib
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    protocol = json.loads((study/'protocol.json').read_text())
    rows = read_rows(study/'inputs.jsonl.gz')
    if len(rows) != protocol['rows']:
        raise ValueError('Incomplete input corpus')
    output = study/'actual'
    output.mkdir(exist_ok=True)
    states = output/'states'
    states.mkdir(exist_ok=True)
    identity_path = output/'run-identity.json'
    identity = dict(source_freeze_sha256=freeze, input_sha256=digest(study/'inputs.jsonl.gz'))
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Existing extraction belongs to another source freeze')
    else:
        if any(states.iterdir()):
            raise ValueError('Unidentified existing state directory')
        identity_path.write_text(json.dumps(identity, indent=2)+'\n')
    if (output/'report.json').exists():
        raise ValueError('Extraction already completed')
    mx.set_memory_limit(10*1024**3)
    mx.set_cache_limit(256*1024**2)
    model, tokenizer = load(str(args.base), tokenizer_config={'trust_remote_code': False})
    model.set_dtype(mx.float32)
    model.freeze()
    labels = [tokenizer.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    if any(len(v) != 1 for v in labels):
        raise ValueError('Unexpected label tokenization')
    prefix = json.loads((study/'common-prefix.json').read_text())
    if len(prefix) != protocol['common_prefix_tokens']:
        raise ValueError('Common prefix differs')
    prefix_cache = None
    if prefix:
        prefix_cache = make_prompt_cache(model.language_model)
        mx.eval(model.language_model.model(mx.array([prefix]), cache=prefix_cache))
    def capture_entries(encoded):
        if not prefix:
            return capture(model, mx.array(encoded), start=protocol['boundary'])
        if any(row[:len(prefix)] != prefix or len(row) <= len(prefix) for row in encoded):
            raise ValueError('Input does not extend the common prefix')
        return capture_with_prefix(model, mx.array([row[len(prefix):] for row in encoded]), prefix_cache, start=protocol['boundary'])
    def encode(entry):
        values = classifier_tokens(tokenizer, entry['row'], [v[0] for v in labels])
        if hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest() != entry['token_sha256'] or len(values) != entry['token_count']:
            raise ValueError('Runtime prompt differs')
        return values
    def split(state, i):
        return dict(state, hidden=state['hidden'][i:i+1],
            shared={k: tuple(v[i:i+1] for v in pair) for k, pair in state['shared'].items()},
            per_layer=[None if v is None else v[i:i+1] for v in state['per_layer']])
    def atomic_write(path, state, entry):
        pending = path.with_name('pending-'+path.name)
        if pending.exists():
            pending.unlink()
        binding = dict(token_sha256=entry['token_sha256'], source_freeze_sha256=freeze)
        write_state(pending, state, binding)
        pending.replace(path)
        return binding
    # Verify capture, serialization and replay against full-sequence execution
    # using only the same training references selected before this experiment.
    old = read_rows(root/'gemma4-readout-141/inputs.jsonl.gz')
    reference_error = 0.
    for index in protocol['reference_indices']:
        entry = old[index]
        if entry['partition'] != 'train':
            raise ValueError('Only training references permitted')
        encoded = [encode(entry)]
        tokens = mx.array(encoded)
        state = capture_entries(encoded)
        path = output/f'reference-{index:06d}.npz'
        binding = atomic_write(path, state, entry)
        restored = to_mlx(read_state(path, expected_identity=binding))
        actual, expected = replay(model, restored), model.language_model.model(tokens)[:, -1:, :]
        mx.eval(actual, expected)
        if not bool(mx.all(mx.isfinite(actual)).item()) or not bool(mx.all(mx.isfinite(expected)).item()):
            raise ValueError('Nonfinite reference replay')
        error = float((mx.max(mx.abs(actual-expected))/mx.maximum(1., mx.max(mx.abs(expected)))).item())
        reference_error = max(reference_error, error)
        if error > 1e-4:
            raise ValueError('Saved shared-tail representation differs')
        del state, restored, actual, expected
        mx.clear_cache()
    (output/'reference-parity.json').write_text(json.dumps(dict(complete=True, rows=len(protocol['reference_indices']),
        max_relative_error=reference_error, only_training_references=True, source_freeze_sha256=freeze), indent=2)+'\n')
    chunks = []
    started = time.monotonic()
    for start in range(0, len(rows), protocol['chunk_size']):
        stop = min(start+protocol['chunk_size'], len(rows))
        path = output/f'chunk-{start:06d}.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record['source_freeze_sha256'] != freeze or (record['start'], record['stop']) != (start, stop) or len(record['states']) != stop-start:
                raise ValueError('Existing chunk identity differs')
            for index, item in zip(range(start, stop), record['states']):
                if item['index'] != index or item['token_sha256'] != rows[index]['token_sha256'] or digest(states/f'state-{index:06d}.npz') != item['sha256']:
                    raise ValueError('Existing cached state differs')
            chunks.append(dict(start=start, stop=stop, sha256=digest(path), cache_bytes=record['cache_bytes']))
            print('RESUMED', stop, '/', len(rows), flush=True)
            continue
        records = []
        cursor = start
        while cursor < stop:
            end = cursor+1
            while end < stop and end-cursor < protocol['batch_size'] and rows[end]['token_count'] == rows[cursor]['token_count']:
                end += 1
            encoded = [encode(entry) for entry in rows[cursor:end]]
            state = capture_entries(encoded)
            for local, index in enumerate(range(cursor, end)):
                destination = states/f'state-{index:06d}.npz'
                atomic_write(destination, split(state, local), rows[index])
                records.append(dict(index=index, token_sha256=rows[index]['token_sha256'],
                    sha256=digest(destination), bytes=destination.stat().st_size))
            del state
            mx.clear_cache()
            cursor = end
        record = dict(start=start, stop=stop, source_freeze_sha256=freeze, states=records, cache_bytes=sum(r['bytes'] for r in records))
        path.write_text(json.dumps(record, indent=2)+'\n')
        chunks.append(dict(start=start, stop=stop, sha256=digest(path), cache_bytes=record['cache_bytes']))
        print('CACHED', stop, '/', len(rows), 'seconds', round(time.monotonic()-started, 2), flush=True)
    report = dict(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), rows=len(rows),
        chunks=chunks, source_freeze_sha256=freeze, cache_bytes=sum(r['cache_bytes'] for r in chunks),
        this_run_seconds=time.monotonic()-started, peak_mlx_bytes=mx.get_peak_memory(),
        reference_max_relative_error=reference_error, calibration_inferred=False, fresh_test_inferred=False,
        synthetic_only=True, independently_annotated=False, stable_release_ready=False)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('FINISHED', json.dumps({k: v for k, v in report.items() if k != 'chunks'}), flush=True)


if __name__ == '__main__':
    main()
