"""Add pre-existing matched TRAIN inputs to the fixed semantic-query view."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_attention import capture_with_prefix, replay
from roguard.gemma4_review import _Engine
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.extract_base_reference_v10 import without_adapters
from training.native_fields_v9 import targets
from training.probe_assertion_query_v10 import tokens_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'candidate', 'query_study', 'query_features', 'scope_study', 'scope_features', 'training', 'study', 'private'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite scope query representations')
    for study, features, expected in ((a.query_study, a.query_features, 'roguard-query-reference-features-1'),
                                      (a.scope_study, a.scope_features, 'roguard-scope-reference-features-1')):
        report = json.loads((study/'report.json').read_text())
        if not report['complete'] or json.loads((study/'protocol.json').read_text())['format'] != expected:
            raise ValueError('Incomplete or incompatible representation parent')
        for lang, record in report['languages'].items():
            for suffix, sha in record['artifacts'].items():
                if file_sha(features/(lang+'-'+suffix)) != sha:
                    raise ValueError('Parent array binding differs')
    audit = json.loads((a.training/'report.json').read_text())
    if file_sha(a.training/'train.jsonl') != audit['training_data_sha256'] or audit['exact_protected_inference_overlap'] != 0:
        raise ValueError('Screened scope inputs differ')
    rows = read_jsonl(a.training/'train.jsonl')
    if len(rows) != 3456 or any(r['partition'] != 'train' for r in rows):
        raise ValueError('Unexpected scope TRAIN population')
    prompt = json.loads((a.query_study/'protocol.json').read_text())['prompt']
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    protocol = {'format': 'roguard-scope-query-features-1', 'source_sha256': file_sha(Path(__file__)),
        'query_parent_report_sha256': file_sha(a.query_study/'report.json'),
        'scope_parent_report_sha256': file_sha(a.scope_study/'report.json'),
        'training_audit_report_sha256': file_sha(a.training/'report.json'),
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-31.json'),
        'prompt': prompt, 'views': ['published-v9', 'zero-adapter-semantic-query'],
        'view_dimensions': 2560, 'calibration_or_test_loaded': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    started = time.monotonic(); report = {'complete': False, 'languages': {}, 'stable_release_ready': False}
    for lang in ('ro', 'uk'):
        old_rows = read_jsonl(a.query_features/(lang+'-rows.jsonl'))
        scope_rows = read_jsonl(a.scope_features/(lang+'-rows.jsonl'))
        lookup = {r['case_id']: i for i, r in enumerate(scope_rows)}
        if len(lookup) != len(scope_rows):
            raise ValueError('Duplicate scope parent identity')
        old_x = np.load(a.query_features/(lang+'-features.npy'), mmap_mode='r')
        old_z = np.load(a.query_features/(lang+'-logits.npy'), mmap_mode='r')
        scope_x = np.load(a.scope_features/(lang+'-features.npy'), mmap_mode='r')
        scope_z = np.load(a.scope_features/(lang+'-logits.npy'), mmap_mode='r')
        added = [r for r in rows if r['language'] == lang]; total = len(old_rows)+len(added)
        x = np.lib.format.open_memmap(a.private/(lang+'-features.npy'), mode='w+', dtype=np.float32, shape=(total, 5120))
        z = np.lib.format.open_memmap(a.private/(lang+'-logits.npy'), mode='w+', dtype=np.float32, shape=(total, 19))
        x[:len(old_rows)] = old_x; z[:len(old_rows)] = old_z
        engine = _Engine(*frozen.models[lang]._args); mx = engine.mx
        output = list(old_rows); seen = {r['query_token_sha256'] for r in old_rows}; sequences = []
        for row in added:
            index = lookup[row['case_id']]; previous = scope_rows[index]
            if previous['partition'] != 'train' or previous['fields'] != targets([row])[0] or previous['gold'] != row['review']:
                raise ValueError('Scope TRAIN labels or identity differ')
            tokens, _ = tokens_for(engine.tokenizer, prompt, {'language': lang, 'turns': row['turns'][:row['focus']+1]})
            identity = digest(tokens)
            if not tokens or len(tokens) > 1024 or identity in seen:
                raise ValueError('Duplicate, empty or unsupported scope query')
            seen.add(identity); sequences.append(tokens)
            x[len(output), :2560] = scope_x[index, :2560]; z[len(output)] = scope_z[index]
            output.append({**previous, 'query_token_sha256': identity})
        common = 0
        for values in zip(*sequences):
            if len(set(values)) != 1:
                break
            common += 1
        common = min(common, min(map(len, sequences))-1)
        if common < 1:
            raise ValueError('Missing query prefix')
        groups = defaultdict(list)
        for i, tokens in enumerate(sequences):
            groups[len(tokens)].append(i)
        batches = [indices[start:start+4] for indices in groups.values() for start in range(0, len(indices), 4)]
        prefix = engine.make_cache(engine.model.base); maximum_error = 0.; done = 0; last_progress = 0
        with without_adapters(engine.model.base):
            mx.eval(engine.model.base.model(mx.array([sequences[0][:common]]), cache=prefix))
            for indices in batches:
                state = capture_with_prefix(engine.model.base, mx.array([sequences[i][common:] for i in indices]), prefix, start=24)
                hidden = replay(engine.model.base, state)[:, 0, :]; mx.eval(hidden); values = np.asarray(hidden)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite query representation')
                x[[len(old_rows)+i for i in indices], 2560:] = values
                if done == 0:
                    full = engine.model.base.model(mx.array([sequences[indices[0]]]))[:, -1, :]; mx.eval(full)
                    maximum_error = float(np.max(np.abs(np.asarray(full)[0]-values[0])))
                    if maximum_error > 1e-3:
                        raise ValueError('Scope query replay differs from full forward')
                    del full
                del hidden, state, values; mx.clear_cache(); done += len(indices)
                if done-last_progress >= 100 or done == len(added):
                    x.flush(); z.flush(); last_progress = done
                    print(json.dumps({'language': lang, 'done': done, 'total': len(added), 'seconds': time.monotonic()-started}), flush=True)
        x.flush(); z.flush()
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError('Invalid output arrays')
        del x, z, old_x, old_z, scope_x, scope_z
        write_jsonl(a.private/(lang+'-rows.jsonl'), output)
        report['languages'][lang] = {'rows': len(output), 'new_training_rows': len(added),
            'partitions': {s: sum(r['partition'] == s for r in output) for s in ('train', 'dev')},
            'query_full_forward_maximum_error': maximum_error, 'duplicate_query_inputs': 0,
            'artifacts': {suffix: file_sha(a.private/(lang+'-'+suffix)) for suffix in ('features.npy', 'logits.npy', 'rows.jsonl')}}
        (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        del engine, prefix
        import gc
        gc.collect(); mx.clear_cache()
    report.update(complete=True, seconds=time.monotonic()-started)
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
