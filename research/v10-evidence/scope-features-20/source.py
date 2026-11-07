"""Append screened curriculum TRAIN features to the frozen paired-view corpus."""
import argparse
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_attention import capture_with_prefix, replay
from roguard.gemma4_contract import classifier_tokens, PREFIX_TOKENS
from roguard.gemma4_review import _Engine
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.extract_base_reference_v10 import without_adapters
from training.native_fields_v9 import targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'candidate', 'parent_features', 'parent_study', 'training', 'study', 'private'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite scope representations')
    parent = json.loads((a.parent_study/'report.json').read_text())
    if not parent['complete'] or json.loads((a.parent_study/'protocol.json').read_text())['format'] != 'roguard-base-reference-features-1':
        raise ValueError('Expected completed paired-view parent features')
    for lang, record in parent['languages'].items():
        for suffix, expected in record['artifacts'].items():
            if file_sha(a.parent_features/(lang+'-'+suffix)) != expected:
                raise ValueError('Parent representation binding differs')
    audit = json.loads((a.training/'report.json').read_text())
    if audit['format'] != 'roguard-scope-input-audit-1' or file_sha(a.training/'train.jsonl') != audit['training_data_sha256']:
        raise ValueError('Screened curriculum binding differs')
    rows = read_jsonl(a.training/'train.jsonl')
    if any(r['partition'] != 'train' for r in rows):
        raise ValueError('Only new TRAIN inputs are permitted')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    protocol = {'format': 'roguard-scope-reference-features-1', 'source_sha256': file_sha(Path(__file__)),
        'parent_report_sha256': file_sha(a.parent_study/'report.json'),
        'input_audit_report_sha256': file_sha(a.training/'report.json'),
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-17.json'),
        'input_clarification_sha256': file_sha('research/v10-evidence/protocol-amendment-18.json'),
        'reference_extractor_source_sha256': file_sha('training/extract_base_reference_v10.py'),
        'views': ['published-v9', 'zero-adapter-base'], 'view_dimensions': 2560,
        'calibration_or_test_loaded': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    started = time.monotonic(); report = {'complete': False, 'languages': {},
        'calibration_or_test_loaded': False, 'stable_release_ready': False}
    for lang in ('ro', 'uk'):
        old_rows = read_jsonl(a.parent_features/(lang+'-rows.jsonl'))
        old_x = np.load(a.parent_features/(lang+'-features.npy'), mmap_mode='r')
        old_z = np.load(a.parent_features/(lang+'-logits.npy'), mmap_mode='r')
        if old_x.shape != (len(old_rows), 5120) or old_z.shape != (len(old_rows), 19):
            raise ValueError('Parent tensor shape differs')
        added = [r for r in rows if r['language'] == lang]; total = len(old_rows)+len(added)
        x = np.lib.format.open_memmap(a.private/(lang+'-features.npy'), mode='w+', dtype=np.float32, shape=(total, 5120))
        z = np.lib.format.open_memmap(a.private/(lang+'-logits.npy'), mode='w+', dtype=np.float32, shape=(total, 19))
        x[:len(old_rows)] = old_x; z[:len(old_rows)] = old_z
        output = [{**r, 'curriculum': False} for r in old_rows]
        seen = {r['token_sha256'] for r in old_rows}
        engine = _Engine(*frozen.models[lang]._args); mx = engine.mx
        common = engine.make_cache(engine.model.base)
        mx.eval(engine.model.base.model(mx.array([PREFIX_TOKENS]), cache=common))
        maximum_full_parity = 0.
        for i, row in enumerate(added):
            payload = {'language': lang, 'turns': row['turns'][:row['focus']+1]}
            tokens = classifier_tokens(engine.tokenizer, payload); identity = digest(tokens)
            if identity in seen:
                raise ValueError('New model-token prefix overlaps existing TRAIN/DEV or another new row')
            if not len(PREFIX_TOKENS) < len(tokens) <= 1024 or tokens[:len(PREFIX_TOKENS)] != PREFIX_TOKENS:
                raise ValueError('Unsupported curriculum token envelope')
            seen.add(identity)
            state = capture_with_prefix(engine.model.base, mx.array([tokens[len(PREFIX_TOKENS):]]), common, start=24)
            adapted = replay(engine.model.base, state)[:, 0, :]
            logits = engine.model.heads[lang](adapted); mx.eval(adapted, logits)
            x[len(old_rows)+i, :2560] = np.asarray(adapted)[0]; z[len(old_rows)+i] = np.asarray(logits)[0]
            with without_adapters(engine.model.base):
                base = replay(engine.model.base, state)[:, 0, :]; mx.eval(base)
                x[len(old_rows)+i, 2560:] = np.asarray(base)[0]
                if i == 0:
                    full = engine.model.base.model(mx.array([tokens]))[:, -1, :]; mx.eval(full)
                    maximum_full_parity = float(np.max(np.abs(np.asarray(full)-np.asarray(base))))
                    if maximum_full_parity > 1e-3:
                        raise ValueError('Fresh scope base features differ from full inference')
                    del full
            output.append({'case_id': row['case_id'], 'language': lang, 'partition': 'train',
                'group': 'scope-curriculum', 'gold': row['review'], 'fields': targets([row])[0],
                'token_sha256': identity, 'family': row['family'], 'curriculum': True})
            del state, adapted, logits, base; mx.clear_cache()
            if (i+1) % 100 == 0 or i+1 == len(added):
                x.flush(); z.flush()
                progress = {'language': lang, 'done': i+1, 'total': len(added), 'seconds': time.monotonic()-started}
                (a.study/'progress.json').write_text(json.dumps(progress, indent=2)+'\n'); print(json.dumps(progress), flush=True)
        x.flush(); z.flush()
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError('Nonfinite curriculum representation')
        del x, z, old_x, old_z
        write_jsonl(a.private/(lang+'-rows.jsonl'), output)
        report['languages'][lang] = {'rows': len(output), 'dimensions': 5120, 'new_training_rows': len(added),
            'partitions': {s: sum(r['partition'] == s for r in output) for s in ('train', 'dev')},
            'base_full_forward_maximum_error': maximum_full_parity,
            'artifacts': {suffix: file_sha(a.private/(lang+'-'+suffix)) for suffix in ('features.npy', 'logits.npy', 'rows.jsonl')}}
        (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        del engine, common
        import gc
        gc.collect(); mx.clear_cache()
    report.update(complete=True, seconds=time.monotonic()-started)
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
