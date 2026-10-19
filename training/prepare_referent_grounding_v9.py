"""Freeze new synthetic focus/denial training and disjoint development probes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.referent_grounding_v9 import generate
from training.gemma4_prompt_v9 import classifier_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    cache = root/'shared-tail-corpus-164'
    manifest = json.loads((cache/'base-manifest.json').read_text())
    if args.base.name != manifest['revision']:
        raise ValueError('Wrong pinned tokenizer revision')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        if digest(args.base/name) != manifest['files'][name]['sha256']:
            raise ValueError('Pinned tokenizer differs')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    labels = [tokenizer.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    if any(len(v) != 1 for v in labels):
        raise ValueError('Unexpected label encoding')
    prefix = json.loads((cache/'common-prefix.json').read_text())
    def identity(row):
        tokens = classifier_tokens(tokenizer, row, [v[0] for v in labels])
        if not len(prefix) < len(tokens) <= 1024 or tokens[:len(prefix)] != prefix:
            raise ValueError('No empty, truncated or differently prefixed inputs')
        return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(), len(tokens)
    old = read_rows(cache/'inputs.jsonl.gz') + read_rows(root/'focus-grounding-173/inputs.jsonl.gz')
    protected = {r['token_sha256'] for r in old}
    sources = {str(p): digest(p) for p in (cache/'inputs.jsonl.gz', root/'focus-grounding-173/inputs.jsonl.gz')}
    for language in ('ro', 'uk'):
        paths = [root/'native-scope-operators-131'/language/(part+'.jsonl.gz') for part in ('cal', 'exposed')]
        paths.append(root/'paired-scope-117/data'/f'{language}-test.jsonl')
        for path in paths:
            sources[str(path)] = digest(path)
            for row in read_rows(path):
                protected.add(identity(dict(row, focus=row.get('focus', len(row['turns'])-1)))[0])
    entries, seen, ids = [], set(), set()
    for language in ('ro', 'uk'):
        for partition in ('train', 'dev'):
            for row in generate(language, partition):
                key, length = identity(row)
                if key in protected or key in seen or row['case_id'] in ids:
                    raise ValueError('Duplicate new input or overlap with a protected population')
                seen.add(key)
                ids.add(row['case_id'])
                entries.append(dict(row=row, partition=partition, group='referent_dev' if partition == 'dev' else 'train',
                    token_sha256=key, token_count=length))
    entries.sort(key=lambda r: (r['token_count'], r['row']['case_id']))
    study = root/'referent-grounding-175'
    study.mkdir(exist_ok=False)
    (study/'inputs.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in entries).encode(), mtime=0))
    protocol = dict(study=175, created_utc=datetime.now(timezone.utc).isoformat(),
        language_partitions=dict(Counter(r['row']['language']+'_'+r['partition'] for r in entries)),
        rows=len(entries), maximum_tokens=max(r['token_count'] for r in entries), protected_token_overlap=0,
        design='Unspecified personal framing versus explicit concern; pronouns inherit only a present concern referent',
        development='Disjoint framing phrases, neutral activities and concern histories',
        informed_by='Study170 Ukrainian conversational false-alert categories: unspecified personal framing and benign referents',
        failed_case_wording_copied=False, synthetic_only=True, independently_annotated=False,
        protected_access='Token identity checks only; no model scores or neural inference',
        calibration_inferred=False, fresh_test_inferred=False, source_input_sha256=sources)
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    paths = [study/'protocol.json', study/'inputs.jsonl.gz', cache/'base-manifest.json', cache/'common-prefix.json']
    paths += [Path('training')/name for name in ('referent_grounding_v9.py', 'prepare_referent_grounding_v9.py',
        'gemma4_prompt_v9.py', 'prepare_instruction_sft_v8.py', 'evaluate_gemma_review_v9.py', 'probe_instruction_v8.py')]
    sources.update({str(p): digest(p) for p in paths})
    (study/'source-freeze.json').write_text(json.dumps(sources, indent=2)+'\n')
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    main()
