"""Extract fixed semantic-query views from original TRAIN/DEV inputs only."""
import argparse
from collections import defaultdict
import gzip
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
from training.prepare_instruction_sft_v8 import target
from training.probe_assertion_query_v10 import PROMPTS, tokens_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'candidate', 'parent_features', 'parent_study', 'study', 'private'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite query representations')
    parent = json.loads((a.parent_study/'report.json').read_text())
    if not parent['complete'] or json.loads((a.parent_study/'protocol.json').read_text())['format'] != 'roguard-base-reference-features-1':
        raise ValueError('Expected completed paired-view parent features')
    for lang, record in parent['languages'].items():
        for suffix, expected in record['artifacts'].items():
            if file_sha(a.parent_features/(lang+'-'+suffix)) != expected:
                raise ValueError('Parent representation binding differs')
    entries = []; inputs = {}
    for name in ('shared-tail-corpus-164', 'grounding-tail-corpus-174', 'referent-tail-corpus-176', 'reported-tail-corpus-195'):
        path = Path('research/v9-stabilization')/name/'inputs.jsonl.gz'
        inputs[str(path)] = file_sha(path)
        entries.extend(map(json.loads, gzip.open(path, 'rt', encoding='utf-8')))
    if any(e['partition'] not in ('train', 'dev') for e in entries):
        raise ValueError('Only original TRAIN and DEV are permitted')
    if {s: sum(e['partition'] == s for e in entries) for s in ('train', 'dev')} != {'train': 16641, 'dev': 6341}:
        raise ValueError('Original population differs')
    prompt = PROMPTS['outer-claim-scope']
    registered = json.loads(Path('research/v10-evidence/assertion-query-22/protocol.json').read_text())
    if prompt != registered['prompts']['outer-claim-scope']:
        raise ValueError('Registered query prompt differs')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    protocol = {'format': 'roguard-query-reference-features-1', 'source_sha256': file_sha(Path(__file__)),
        'query_helper_sha256': file_sha('training/probe_assertion_query_v10.py'),
        'parent_report_sha256': file_sha(a.parent_study/'report.json'), 'inputs': inputs,
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-22.json'),
        'prompt': prompt, 'views': ['published-v9', 'zero-adapter-semantic-query'], 'view_dimensions': 2560,
        'calibration_or_test_loaded': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'query-source.py').write_bytes(Path('training/probe_assertion_query_v10.py').read_bytes())
    import numpy as np
    started = time.monotonic(); report = {'complete': False, 'languages': {},
        'calibration_or_test_loaded': False, 'stable_release_ready': False}
    for lang in ('ro', 'uk'):
        selected = [e for e in entries if e['row']['language'] == lang]
        old_rows = read_jsonl(a.parent_features/(lang+'-rows.jsonl'))
        lookup = {r['token_sha256']: i for i, r in enumerate(old_rows)}
        if len(lookup) != len(old_rows):
            raise ValueError('Duplicate parent token identity')
        old_x = np.load(a.parent_features/(lang+'-features.npy'), mmap_mode='r')
        old_z = np.load(a.parent_features/(lang+'-logits.npy'), mmap_mode='r')
        engine = _Engine(*frozen.models[lang]._args); mx = engine.mx
        sequences = []; output = []; seen = set()
        for e in selected:
            row = e['row']; payload = {'language': lang, 'turns': row['turns'][:row['focus']+1]}
            tokens, _ = tokens_for(engine.tokenizer, prompt, payload); identity = digest(tokens)
            if not tokens or len(tokens) > 1024 or identity in seen:
                raise ValueError('Invalid, oversized or duplicate query input')
            seen.add(identity); sequences.append(tokens)
            previous = old_rows[lookup[e['token_sha256']]]
            if previous['case_id'] != row['case_id'] or previous['partition'] != e['partition'] or previous['fields'] != targets([row])[0] or previous['gold'] != target(row):
                raise ValueError('Parent row identity differs')
            output.append({**previous, 'query_token_sha256': identity})
        common = 0
        for values in zip(*sequences):
            if len(set(values)) != 1:
                break
            common += 1
        common = min(common, min(map(len, sequences))-1)
        if common < 1:
            raise ValueError('No shared query prefix')
        x = np.lib.format.open_memmap(a.private/(lang+'-features.npy'), mode='w+', dtype=np.float32, shape=(len(selected), 5120))
        indices = [lookup[e['token_sha256']] for e in selected]
        x[:, :2560] = old_x[indices, :2560]
        np.save(a.private/(lang+'-logits.npy'), old_z[indices], allow_pickle=False)
        parity = {}; prefix = engine.make_cache(engine.model.base)
        groups = defaultdict(list)
        for i, tokens in enumerate(sequences):
            groups[len(tokens)].append(i)
        batches = [indices[start:start+4] for indices in groups.values() for start in range(0, len(indices), 4)]
        done = 0; last_progress = 0
        with without_adapters(engine.model.base):
            mx.eval(engine.model.base.model(mx.array([sequences[0][:common]]), cache=prefix))
            for chosen in batches:
                state = capture_with_prefix(engine.model.base, mx.array([sequences[i][common:] for i in chosen]), prefix, start=24)
                hidden = replay(engine.model.base, state)[:, 0, :]; mx.eval(hidden)
                values = np.asarray(hidden)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite query representation')
                x[chosen, 2560:] = values
                for local, i in enumerate(chosen):
                    partition = selected[i]['partition']
                    if partition not in parity:
                        full = engine.model.base.model(mx.array([sequences[i]]))[:, -1, :]; mx.eval(full)
                        parity[partition] = float(np.max(np.abs(np.asarray(full)[0]-values[local])))
                        if parity[partition] > 1e-3:
                            raise ValueError('Batched query replay differs from full inference')
                        del full
                del hidden, state, values; mx.clear_cache()
                done += len(chosen)
                if done-last_progress >= 100 or done == len(selected):
                    last_progress = done
                    x.flush()
                    progress = {'language': lang, 'done': done, 'total': len(selected), 'seconds': time.monotonic()-started}
                    (a.study/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                    print(json.dumps(progress), flush=True)
        x.flush(); del x, old_x, old_z
        write_jsonl(a.private/(lang+'-rows.jsonl'), output)
        report['languages'][lang] = {'rows': len(output), 'dimensions': 5120,
            'partitions': {s: sum(r['partition'] == s for r in output) for s in ('train', 'dev')},
            'query_full_forward_maximum_error': parity, 'common_prefix_tokens': common,
            'maximum_equal_length_batch': 4,
            'duplicate_query_inputs': 0, 'train_dev_query_overlap': 0,
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
