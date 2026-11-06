"""Extract paired frozen v9/base representations using verified shared-KV states."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.gemma4_attention import replay, _backbone
from roguard.gemma4_contract import classifier_tokens
from roguard.gemma4_review import _Engine, _score
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.native_fields_v9 import targets
from training.prepare_instruction_sft_v8 import target
from training.shared_tail_cache_v9 import batch_states, to_mlx
from training.train_semantic_tail_v10 import MixedReader


@contextmanager
def without_adapters(model):
    """Zero only task-adapter output matrices; restore even on failed inference."""
    import mlx.core as mx
    body = _backbone(model, 24)
    saved = [(getattr(layer.self_attn, name), getattr(layer.self_attn, name).lora_b)
             for layer in body.layers[24:] for name in ('q_proj', 'o_proj')]
    if len(saved) != 36:
        raise ValueError('Unexpected adapter inventory')
    try:
        for module, value in saved:
            module.lora_b = mx.zeros_like(value)
        yield
    finally:
        for module, value in saved:
            module.lora_b = value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'candidate', 'added_study', 'added_private', 'development_study',
                'development_private', 'features', 'feature_study', 'study', 'private'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('research/v9-stabilization'))
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite paired representations')
    original = json.loads((a.feature_study/'report.json').read_text())
    if not original['complete']:
        raise ValueError('Original representations incomplete')
    for lang, record in original['languages'].items():
        for suffix, expected in record['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix)) != expected:
                raise ValueError('Original representation binding differs')
    reader = MixedReader(a.added_study, a.added_private, a.development_study,
                         a.development_private, a.cache_root)
    if any(r['partition'] not in ('train', 'dev') for r in reader.entries):
        raise ValueError('Only TRAIN and DEV may be extracted')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    protocol = {'format': 'roguard-base-reference-features-1',
        'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-15.json'),
        'original_feature_report_sha256': file_sha(a.feature_study/'report.json'),
        'added_cache_report_sha256': file_sha(a.added_study/'report.json'),
        'development_cache_report_sha256': file_sha(a.development_study/'report.json'),
        'old_cache_parents': reader.old.parents,
        'views': ['published-v9', 'zero-adapter-base'], 'view_dimensions': 2560,
        'calibration_or_test_loaded': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    started = time.monotonic()
    report = {'complete': False, 'languages': {}, 'calibration_or_test_loaded': False,
              'stable_release_ready': False}
    for lang in ('ro', 'uk'):
        component = frozen.models[lang]; engine = _Engine(*component._args); mx = engine.mx
        indices = [i for i, r in enumerate(reader.entries) if r['row']['language'] == lang]
        entries = [reader.entries[i] for i in indices]
        teacher_rows = read_jsonl(a.features/(lang+'-rows.jsonl'))
        teacher = np.load(a.features/(lang+'-logits.npy'), mmap_mode='r')
        lookup = {r['token_sha256']: i for i, r in enumerate(teacher_rows)}
        features = np.lib.format.open_memmap(a.private/(lang+'-features.npy'), mode='w+',
                     dtype=np.float32, shape=(len(indices), 5120))
        logits = np.empty((len(indices), 19), dtype=np.float32)
        maximum_score_error = 0.; base_parity = None
        chunks = [indices[i:i+64] for i in range(0, len(indices), 64)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(reader.read_many, chunks[0]); offset = 0
            for block, chosen in enumerate(chunks):
                states = future.result()
                if block+1 < len(chunks):
                    future = pool.submit(reader.read_many, chunks[block+1])
                groups = defaultdict(list)
                for i, state in enumerate(states):
                    groups[state['length']].append(i)
                for group in groups.values():
                    for start in range(0, len(group), 8):
                        local = group[start:start+8]
                        state = batch_states([to_mlx(states[i]) for i in local])
                        adapted = replay(engine.model.base, state)[:, 0, :]
                        z = engine.model.heads[lang](adapted); mx.eval(adapted, z)
                        features[[offset+i for i in local], :2560] = np.asarray(adapted)
                        logits[[offset+i for i in local]] = np.asarray(z)
                        for j, i in enumerate(local):
                            entry = entries[offset+i]
                            reference = entry.get('reference_logits')
                            if reference is None:
                                record = teacher_rows[lookup[entry['token_sha256']]]
                                if record['partition'] != entry['partition'] or record['fields'] != targets([entry['row']])[0]:
                                    raise ValueError('Original teacher row binding differs')
                                reference = teacher[lookup[entry['token_sha256']]]
                            difference = abs(_score(np.asarray(z)[j], component._spec)-_score(reference, component._spec))
                            maximum_score_error = max(maximum_score_error, difference)
                            if difference > 1e-5:
                                raise ValueError('Published v9 replay score differs')
                        with without_adapters(engine.model.base):
                            base = replay(engine.model.base, state)[:, 0, :]; mx.eval(base)
                            features[[offset+i for i in local], 2560:] = np.asarray(base)
                            if base_parity is None:
                                row = entries[offset+local[0]]['row']
                                if entries[offset+local[0]]['partition'] != 'train':
                                    raise ValueError('Parity sample must be TRAIN')
                                payload = {'language': lang, 'turns': row['turns'][:row['focus']+1]}
                                tokens = classifier_tokens(engine.tokenizer, payload)
                                full = engine.model.base.model(mx.array([tokens]))[:, -1, :]; mx.eval(full)
                                base_parity = float(np.max(np.abs(np.asarray(full)[0]-np.asarray(base)[0])))
                                if base_parity > 1e-3:
                                    raise ValueError('Zero-adapter full and cached inference differ')
                                del full
                        if not np.isfinite(np.asarray(base)).all() or not np.isfinite(np.asarray(adapted)).all():
                            raise ValueError('Nonfinite representation')
                        del state, base, adapted, z
                offset += len(chosen); del states; mx.clear_cache()
                if block % 20 == 0 or offset == len(indices):
                    progress = {'language': lang, 'done': offset, 'total': len(indices),
                                'seconds': time.monotonic()-started}
                    (a.study/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                    print(json.dumps(progress), flush=True)
        features.flush(); del features
        np.save(a.private/(lang+'-logits.npy'), logits, allow_pickle=False)
        rows = [{'case_id': e['row']['case_id'], 'language': lang, 'partition': e['partition'],
                 'group': e['group'], 'gold': target(e['row']), 'fields': targets([e['row']])[0],
                 'token_sha256': e['token_sha256']} for e in entries]
        write_jsonl(a.private/(lang+'-rows.jsonl'), rows)
        report['languages'][lang] = {'rows': len(rows), 'dimensions': 5120,
            'partitions': {s: sum(r['partition'] == s for r in rows) for s in ('train', 'dev')},
            'maximum_v9_score_error': maximum_score_error, 'base_full_forward_maximum_error': base_parity,
            'artifacts': {suffix: file_sha(a.private/(lang+'-'+suffix))
                          for suffix in ('features.npy', 'logits.npy', 'rows.jsonl')}}
        (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        del engine, teacher, logits
        import gc
        gc.collect(); mx.clear_cache()
    report.update(complete=True, seconds=time.monotonic()-started)
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
