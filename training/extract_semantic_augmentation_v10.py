"""Append frozen-v9 representations of screened training-only augmentations."""
import argparse
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from roguard.trajectory_neural import FieldEngine
from training.native_fields_v9 import targets


class CachedEngine(FieldEngine):
    cache_instructions = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'candidate', 'training', 'features', 'feature_study', 'study', 'private'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite features')
    audit = json.loads((a.training/'report.json').read_text())
    if file_sha(a.training/'train.jsonl') != audit['training_data_sha256']:
        raise ValueError('Training dataset binding differs')
    extraction = json.loads((a.feature_study/'report.json').read_text())
    if not extraction['complete']:
        raise ValueError('Original extraction is incomplete')
    for lang, record in extraction['languages'].items():
        for suffix, expected in record['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix)) != expected:
                raise ValueError('Original feature binding differs')
    rows = read_jsonl(a.training/'train.jsonl')
    if any(r['partition'] != 'train' for r in rows):
        raise ValueError('Only new training inputs are allowed')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    protocol = {'format': 'roguard-semantic-augmentation-features-1',
        'source_sha256': file_sha(Path(__file__)),
        'training_report_sha256': file_sha(a.training/'report.json'),
        'parent_report_sha256': file_sha(a.feature_study/'report.json'),
        'new_partitions': ['train'], 'calibration_or_test_loaded': False,
        'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    import mlx.nn as nn
    from roguard.gemma4_contract import classifier_tokens
    from roguard.assurance import digest
    report = {'complete': False, 'languages': {}}; started = time.monotonic()
    for lang in ('ro', 'uk'):
        component = frozen.models[lang]; engine = CachedEngine(*component._args); mx = engine.mx
        class Capture(nn.Module):
            def __init__(self, inner):
                super().__init__(); self.inner = inner
            def __call__(self, h):
                z = self.inner(h); mx.eval(h, z)
                self.h = np.asarray(h).copy(); self.z = np.asarray(z).copy()
                return z
        tap = Capture(engine.model.heads[lang]); engine.model.heads[lang] = tap
        old_rows = read_jsonl(a.features/(lang+'-rows.jsonl'))
        added = [r for r in rows if r['language'] == lang]; n = len(old_rows)+len(added)
        features = np.lib.format.open_memmap(a.private/(lang+'-features.npy'),
            mode='w+', dtype=np.float32, shape=(n, 2560))
        logits = np.empty((n, 19), dtype=np.float32)
        features[:len(old_rows)] = np.load(a.features/(lang+'-features.npy'), mmap_mode='r')
        logits[:len(old_rows)] = np.load(a.features/(lang+'-logits.npy'), mmap_mode='r')
        metadata = list(old_rows)
        for j, row in enumerate(added):
            payload = {'language': lang, 'turns': row['turns']}
            if len(payload['turns']) > 12 or engine.scores([payload], component._spec)[0] is None:
                raise ValueError('Training input exceeds the frozen encoder envelope')
            features[len(old_rows)+j] = tap.h[0]; logits[len(old_rows)+j] = tap.z[0]
            metadata.append({'case_id': row['case_id'], 'language': lang, 'partition': 'train',
                'group': row['source'], 'gold': row['review'], 'fields': targets([row])[0],
                'token_sha256': digest(classifier_tokens(engine.tokenizer, payload))})
            if (j+1)%100 == 0 or j+1 == len(added):
                features.flush(); progress = {'language': lang, 'done': j+1,
                    'total': len(added), 'seconds': time.monotonic()-started}
                (a.study/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                print(json.dumps(progress), flush=True)
        features.flush(); del features
        np.save(a.private/(lang+'-logits.npy'), logits, allow_pickle=False)
        write_jsonl(a.private/(lang+'-rows.jsonl'), metadata)
        report['languages'][lang] = {'rows': n, 'dimensions': 2560,
            'partitions': {s: sum(r['partition'] == s for r in metadata) for s in ('train', 'dev')},
            'artifacts': {suffix: file_sha(a.private/(lang+'-'+suffix))
                          for suffix in ('features.npy', 'logits.npy', 'rows.jsonl')}}
        (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        del engine, tap
        import gc
        gc.collect()
    report.update(complete=True, seconds=time.monotonic()-started,
                  calibration_or_test_loaded=False, stable_release_ready=False)
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
