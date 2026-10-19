"""Prepare independent synthetic training and development context transitions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from training.context_transition_v9 import generate
from training.diagnose_contextual_complementarity_v9 import read_rows
from training.policy_entailment_v9 import HYPOTHESES
from training.train_paired_nli_v9 import premise


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from roguard.nli_v21 import verify_base
    verify_base(args.base)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    root = Path('research/v9-stabilization')
    study = root/'context-transition-161'
    study.mkdir(exist_ok=False)
    source_hashes = {}
    def tracked(path):
        source_hashes[str(path)] = digest(path)
        return read_rows(path)
    def keys(rows):
        result = []
        for start in range(0, len(rows), 128):
            part = rows[start:start+128]
            encoded = tokenizer([premise(r) for r in part for _ in HYPOTHESES], list(HYPOTHESES)*len(part), padding=False, truncation=False)
            if any(len(v) > 512 for v in encoded['input_ids']):
                raise ValueError('No truncation permitted')
            result.extend(hashlib.sha256(json.dumps(encoded['input_ids'][i:i+2], separators=(',', ':')).encode()).hexdigest() for i in range(0, 2*len(part), 2))
        return result
    protected = []
    for language in ('ro', 'uk'):
        for partition in ('dev', 'scope_dev', 'cal', 'exposed')+(('stress',) if language == 'uk' else ()):
            protected += tracked(root/'reason-coverage-123/candidate'/f'{language}_{partition}.jsonl.gz')
        protected += tracked(root/'paired-scope-117/data'/f'{language}-test.jsonl')
    protected_keys = set(keys(protected))
    generated = {(lang, part): generate(lang, part) for lang in ('ro', 'uk') for part in ('train', 'dev')}
    all_new_keys = set()
    counts = {}
    for (language, partition), rows in generated.items():
        encoded = keys(rows)
        if len(encoded) != len(set(encoded)) or set(encoded) & (protected_keys | all_new_keys):
            raise ValueError('Generated duplicate or protected overlap')
        all_new_keys.update(encoded)
        path = study/f'{language}-{partition}.jsonl.gz'
        path.write_bytes(gzip.compress(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode(), mtime=0))
        counts[f'{language}_{partition}'] = dict(rows=len(rows), labels=dict(Counter(r['review'] for r in rows)), sha256=digest(path))
    protocol = dict(study=161, created_utc=datetime.now(timezone.utc).isoformat(), counts=counts,
        training_only=False, development_design='Disjoint current-message bodies and framing; shared nuisance histories',
        design_informed_by='Exposed158 context-switch error category; no failed-case wording copied',
        synthetic_only=True, independently_annotated=False, protected_token_overlap=0,
        fresh_test_neurally_inferred=False, no_foundation_inference=True,
        source_input_sha256=source_hashes, stable_release_ready=False)
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    paths = list(study.glob('*'))+[Path(p) for p in ('training/context_transition_v9.py', 'training/prepare_context_transition_v9.py',
        'training/train_paired_nli_v9.py', 'training/policy_entailment_v9.py', 'training/diagnose_contextual_complementarity_v9.py',
        'src/roguard/semantic.py', 'src/roguard/nli_v21.py')]
    (study/'source-freeze.json').write_text(json.dumps({str(p): digest(p) for p in paths}, indent=2)+'\n')
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    main()
