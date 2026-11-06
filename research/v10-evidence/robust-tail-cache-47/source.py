"""Cache frozen-body states of screened TRAIN inputs, with exact replay checks."""
import argparse
import json
from pathlib import Path
import shutil
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_attention import capture_with_prefix, replay
from roguard.gemma4_contract import classifier_tokens, PREFIX_TOKENS
from roguard.gemma4_review import _Engine
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.shared_tail_cache_v9 import write_state, read_state, to_mlx


class AddedReader:
    def __init__(self, study, private):
        self.study, self.private = Path(study), Path(private)
        report = json.loads((self.study/'report.json').read_text())
        if not report['complete'] or file_sha(self.study/'inputs.jsonl') != report['inputs_sha256']:
            raise ValueError('Additional tail cache is incomplete or changed')
        self.entries = read_jsonl(self.study/'inputs.jsonl')
        self.records = report['states']; self.binding = report['source_freeze_sha256']
        self.verified = {}
        if len(self.records) != len(self.entries):
            raise ValueError('Incomplete cached states')

    def read_many(self, indices):
        result = []
        for index in indices:
            record = self.records[index]; entry = self.entries[index]
            path = self.private/('state-'+str(index).zfill(6)+'.npz')
            stat = path.stat(); signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
            if self.verified.get(index) != signature:
                if record['index'] != index or stat.st_size != record['bytes'] or file_sha(path) != record['sha256']:
                    raise ValueError('Additional tail state differs')
                self.verified[index] = signature
            result.append(read_state(path, expected_identity={
                'token_sha256': entry['token_sha256'], 'source_freeze_sha256': self.binding}))
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'candidate', 'training', 'features', 'feature_study', 'study', 'private'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite a tail cache')
    audit = json.loads((a.training/'report.json').read_text())
    if file_sha(a.training/'train.jsonl') != audit['training_data_sha256']:
        raise ValueError('Screened training inputs differ')
    features = json.loads((a.feature_study/'report.json').read_text())
    if not features['complete']:
        raise ValueError('Reference representation is incomplete')
    for lang, record in features['languages'].items():
        for suffix, sha in record['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix)) != sha:
                raise ValueError('Reference representation differs')
    rows = read_jsonl(a.training/'train.jsonl')
    if any(r['partition'] != 'train' for r in rows):
        raise ValueError('Only training inputs may be cached here')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    binding = file_sha(a.training/'report.json')
    protocol = {'format': 'roguard-semantic-tail-cache-1', 'source_sha256': file_sha(Path(__file__)),
        'training_audit_sha256': binding, 'reference_feature_report_sha256': file_sha(a.feature_study/'report.json'),
        'base_and_adapter_identities': {k:v.identity for k,v in frozen.models.items()},
        'allowed_partition': 'train', 'minimum_disk_reserve_bytes': 8*1024**3,
        'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    entries = []; records = []; started = time.monotonic(); max_error = 0
    for lang in ('ro', 'uk'):
        component = frozen.models[lang]; engine = _Engine(*component._args); mx = engine.mx
        cache = engine.make_cache(engine.model.base)
        mx.eval(engine.model.base.model(mx.array([PREFIX_TOKENS]), cache=cache))
        reference = np.load(a.features/(lang+'-features.npy'), mmap_mode='r')
        metadata = read_jsonl(a.features/(lang+'-rows.jsonl'))
        lookup = {r['case_id']: i for i,r in enumerate(metadata)}
        for row in (r for r in rows if r['language'] == lang):
            if shutil.disk_usage(a.private).free < 8*1024**3+128*1024**2:
                raise ValueError('Stopping cache extraction to preserve disk reserve; incomplete study retained')
            payload = {'language': lang, 'turns': row['turns']}
            tokens = classifier_tokens(engine.tokenizer, payload)
            if not len(PREFIX_TOKENS) < len(tokens) <= 1024 or tokens[:len(PREFIX_TOKENS)] != PREFIX_TOKENS:
                raise ValueError('Unexpected training prompt or envelope')
            token_sha = digest(tokens)
            state = capture_with_prefix(engine.model.base, mx.array([tokens[len(PREFIX_TOKENS):]]), cache, start=24)
            h = replay(engine.model.base, state)[:, 0, :]; mx.eval(h)
            error = float(np.max(np.abs(np.asarray(h)[0]-reference[lookup[row['case_id']]])))
            max_error = max(max_error, error)
            if error > 1e-5:
                raise ValueError('Frozen-body cache does not replay reference features exactly')
            index = len(entries); path = a.private/('state-'+str(index).zfill(6)+'.npz')
            write_state(path, state, {'token_sha256': token_sha, 'source_freeze_sha256': binding})
            entries.append({'row': row, 'partition': 'train', 'group': row['source'],
                            'token_sha256': token_sha, 'token_count': len(tokens)})
            records.append({'index': index, 'sha256': file_sha(path), 'bytes': path.stat().st_size})
            del state, h; mx.clear_cache()
            if len(entries)%100 == 0 or len(entries) == len(rows):
                progress = {'done': len(entries), 'total': len(rows), 'seconds': time.monotonic()-started,
                            'maximum_feature_replay_error': max_error,
                            'cache_bytes': sum(r['bytes'] for r in records)}
                (a.study/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                print(json.dumps(progress), flush=True)
        del engine, cache, reference
        import gc
        gc.collect()
    write_jsonl(a.study/'inputs.jsonl', entries)
    report = {'complete': True, 'source_freeze_sha256': binding,
              'inputs_sha256': file_sha(a.study/'inputs.jsonl'), 'states': records,
              'maximum_feature_replay_error': max_error, 'seconds': time.monotonic()-started,
              'stable_release_ready': False}
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
