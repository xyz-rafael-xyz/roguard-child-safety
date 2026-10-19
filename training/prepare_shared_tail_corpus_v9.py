"""Freeze an expanded bilingual corpus after actual shared-tail parity passes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.gemma4_prompt_v9 import classifier_tokens
from training.prepare_instruction_sft_v8 import target


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    preflight = root/'shared-tail-preflight-163'
    report = json.loads((preflight/'actual/report.json').read_text())
    if not report.get('complete') or not report['gradient']['norm'] > 0 or report['gradient']['post_update_relative_error'] > 1e-4:
        raise ValueError('Actual pinned-weight tail preflight must pass first')
    if any(r['tails']['24']['relative_feature_error'] > 1e-4 for r in report['records']):
        raise ValueError('Actual full-tail parity failed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    labels = [tokenizer.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    if any(len(v) != 1 for v in labels):
        raise ValueError('Unexpected label tokenization')
    source_hashes = {}
    def tracked(path):
        source_hashes[str(path)] = digest(path)
        return read_rows(path)
    def identity(row, with_tokens=False):
        tokens = classifier_tokens(tokenizer, row, [v[0] for v in labels])
        if not tokens or len(tokens) > 1024:
            raise ValueError('No empty or truncated prompts')
        result = (hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(), len(tokens))
        return (*result, tokens) if with_tokens else result
    entries = tracked(root/'gemma4-readout-141/inputs.jsonl.gz')+tracked(root/'gemma4-expansion-144/inputs.jsonl.gz')
    for language in ('ro', 'uk'):
        for partition in ('train', 'dev'):
            for row in tracked(root/'context-transition-161'/f'{language}-{partition}.jsonl.gz'):
                key, length = identity(row)
                entries.append(dict(row=row, partition=partition, group='transition_dev' if partition == 'dev' else 'train',
                    token_sha256=key, token_count=length))
    keys = [r['token_sha256'] for r in entries]
    if len(keys) != len(set(keys)) or len({r['row']['case_id'] for r in entries}) != len(entries):
        raise ValueError('Duplicate corpus inputs or identities')
    protected = set()
    for language in ('ro', 'uk'):
        for partition in ('cal', 'exposed'):
            for row in tracked(root/'native-scope-operators-131'/language/(partition+'.jsonl.gz')):
                protected.add(identity(dict(row, focus=row.get('focus', len(row['turns'])-1)))[0])
        for row in tracked(root/'paired-scope-117/data'/f'{language}-test.jsonl'):
            protected.add(identity(row)[0])
    if set(keys) & protected:
        raise ValueError('Prior or reserved population overlaps cache corpus')
    prefix = None
    for entry in entries:
        key, length, tokens = identity(entry['row'], with_tokens=True)
        if (key, length) != (entry['token_sha256'], entry['token_count']):
            raise ValueError('Inherited prompt identity changed')
        if prefix is None:
            prefix = tokens
        else:
            common = 0
            for left, right in zip(prefix, tokens):
                if left != right:
                    break
                common += 1
            prefix = prefix[:common]
        if 'review' in entry['row'] and target(entry['row']) != entry['row']['review']:
            raise ValueError('Semantic policy label differs')
    maximum_tokens = max(r['token_count'] for r in entries)
    bytes_per_token = max(r['tails']['24']['cache_bytes']/r['tokens'] for r in report['records'])
    estimated_bytes = len(entries)*(int(bytes_per_token*maximum_tokens)+65536)
    if shutil.disk_usage(root).free < estimated_bytes+20*1024**3:
        raise ValueError('Insufficient local disk for conservative cache estimate and reserve')
    study = root/'shared-tail-corpus-164'
    study.mkdir(exist_ok=False)
    prefix = prefix[:min(511, min(r['token_count'] for r in entries)-1)]
    (study/'common-prefix.json').write_text(json.dumps(prefix)+'\n')
    entries.sort(key=lambda r: (r['token_count'], r['row']['case_id']))
    (study/'inputs.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in entries).encode(), mtime=0))
    for name in ('base-manifest.json', 'software.json'):
        shutil.copyfile(preflight/name, study/name)
    protocol = dict(study=164, created_utc=datetime.now(timezone.utc).isoformat(), boundary=24,
        rows=len(entries), partitions=dict(Counter(r['partition'] for r in entries)),
        language_partitions=dict(Counter(r['row']['language']+'_'+r['partition'] for r in entries)),
        batch_size=4, chunk_size=100, compute_dtype='float32', storage='Typed identity-bound NPZ, no pickle',
        reference_indices=json.loads((preflight/'protocol.json').read_text())['reference_indices'],
        common_prefix_tokens=len(prefix), prefix_optimization='Immutable prompt prefix; full-sequence training-reference parity required before extraction',
        estimated_cache_bytes=estimated_bytes, protected_token_overlap=0, training_development_overlap=0,
        maximum_tokens=maximum_tokens, source_input_sha256=source_hashes,
        calibration_inferred=False, fresh_test_inferred=False, synthetic_only=True,
        independently_annotated=False, stable_release_ready=False)
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    paths = list(study.glob('*'))+[preflight/'source-freeze.json', preflight/'actual/report.json']
    paths += [Path('training')/p for p in ('prepare_shared_tail_corpus_v9.py', 'extract_shared_tail_corpus_v9.py',
        'gemma4_shared_tail_v9.py', 'shared_tail_cache_v9.py', 'gemma4_prompt_v9.py',
        'gemma4_shared_prefix_v9.py',
        'diagnose_contextual_complementarity_v9.py', 'prepare_instruction_sft_v8.py',
        'evaluate_gemma_review_v9.py', 'probe_instruction_v8.py')]
    paths += [Path('training/extract_gemma4_qualified_scope_v9.py')]
    paths += [Path('src/roguard/evidence.py')]
    (study/'source-freeze.json').write_text(json.dumps({str(p): digest(p) for p in paths}, indent=2)+'\n')
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    main()
