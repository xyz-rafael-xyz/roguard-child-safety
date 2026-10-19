"""Freeze training-only scope contrasts after three tokenizer overlap audits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from training.scope_operators_v9 import generate
from training.policy_entailment_v9 import HYPOTHESES
from training.train_paired_nli_v9 import premise
from training.train_bare_distress_v9 import digest, read, texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ro-base', type=Path, required=True)
    parser.add_argument('--uk-base', type=Path, required=True)
    parser.add_argument('--nli-base', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from roguard.nli_v21 import verify_base
    from roguard.native_assets import reject_native_overrides

    output = Path('research/v9-stabilization/scope-operators-130')
    if output.exists():
        raise ValueError('Scope data already prepared; do not overwrite a freeze')
    source = Path('research/v9-stabilization/reason-coverage-123/data')
    protected = []
    inputs = {}

    def tracked(path):
        inputs[str(path)] = digest(path)
        return read(path) if path.name.endswith('.gz') else [json.loads(l) for l in path.read_text().splitlines()]

    for split in ('ro_dev', 'uk_dev', 'ro_scope_dev', 'uk_scope_dev', 'uk_stress',
                  'ro_cal', 'uk_cal', 'ro_exposed', 'uk_exposed'):
        protected += tracked(source / (split+'.jsonl.gz'))
    for language in ('ro', 'uk'):
        protected += tracked(Path('research/v9-stabilization/paired-scope-117/data') / (language+'-test.jsonl'))
    existing = tracked(source / 'train.jsonl.gz')
    for language in ('ro', 'uk'):
        existing += tracked(Path('research/v9-stabilization/morphology-coverage-127') / (language+'-train.jsonl.gz'))
    authored = {language: generate(language) for language in ('ro', 'uk')}
    audits = {}
    inventory_path = Path('research/v9-stabilization/native-full-22-base-manifest.json')
    inventory = json.loads(inventory_path.read_text())['models']
    inputs[str(inventory_path)] = digest(inventory_path)
    for kind in ('ro', 'uk', 'nli'):
        base = getattr(args, kind+'_base')
        if kind == 'nli':
            verify_base(base)
            selected = lambda rows: rows
        else:
            entry = inventory[kind]
            if base.name != entry['revision']:
                raise ValueError('Native base revision differs')
            reject_native_overrides(base, entry['files'])
            for name, details in entry['files'].items():
                if digest(base/name) != details['sha256']:
                    raise ValueError('Native base file differs')
            selected = lambda rows: [r for r in rows if r['language'] == kind]
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)

        def keys(rows):
            result = []
            maximum = 0
            for start in range(0, len(rows), 128):
                batch = rows[start:start+128]
                if kind == 'nli':
                    tokenized = tokenizer([premise(r) for r in batch for _ in HYPOTHESES],
                                          list(HYPOTHESES)*len(batch), padding=False, truncation=False)
                else:
                    pairs = [texts(r) for r in batch]
                    tokenized = tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
                                          padding=False, truncation=False)
                ids = tokenized['input_ids']
                maximum = max(maximum, max(map(len, ids)))
                if maximum > 512:
                    raise ValueError('No silent truncation permitted')
                result.extend(hashlib.sha256(json.dumps(v, separators=(',', ':')).encode()).hexdigest() for v in ids)
            return result, maximum

        new, maximum = keys(selected(authored['ro']+authored['uk']))
        protected_keys, _ = keys(selected(protected))
        existing_keys, _ = keys(selected(existing))
        overlaps = dict(protected=len(set(new)&set(protected_keys)), existing_training=len(set(new)&set(existing_keys)))
        if any(overlaps.values()) or len(set(new)) != len(new):
            raise ValueError('Unexpected new-input overlap: '+json.dumps(overlaps))
        audits[kind] = dict(unique_inputs=len(new), maximum_tokens=maximum, overlaps=overlaps)
        print(kind, json.dumps(audits[kind]), flush=True)
    output.mkdir()
    for language, rows in authored.items():
        (output/(language+'-train.jsonl.gz')).write_bytes(gzip.compress(
            ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode(), mtime=0))
    protocol = dict(study=130, created_utc=datetime.now(timezone.utc).isoformat(),
                    status='training-data-prepared-only', training_not_started=True,
                    content='Study117 training content only; newly authored scope wording',
                    motivation='Completed123 development errors: RO reported11/reset4; UK denial12/reported9/fiction8',
                    rows_per_language=1920, paired_rows_per_language=1344, ordinary_rows_per_language=576,
                    training_label_counts={l:dict(Counter(str(r['review']) for r in rows)) for l, rows in authored.items()},
                    protects='All nine current prior evaluation splits and the reserved600; no new neural inference',
                    synthetic_only=True, independently_annotated=False, current_frozen_experiments_unchanged=True,
                    release_gates_unchanged=True)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (output/'token-audit.json').write_text(json.dumps(dict(audits=audits, input_sha256=inputs,
        neural_inference_performed=False), indent=2)+'\n')
    files = [Path('training/prepare_scope_operators_v9.py'), Path('training/scope_operators_v9.py'),
             Path('training/paired_scope_data_v9.py'), Path('training/train_bare_distress_v9.py'),
             Path('training/train_paired_nli_v9.py'), Path('training/policy_entailment_v9.py'),
             Path('src/roguard/semantic.py'), Path('src/roguard/context.py'),
             *sorted(output.glob('*.json')), *sorted(output.glob('*.gz'))]
    (output/'source-freeze.json').write_text(json.dumps({str(p):digest(p) for p in files}, indent=2)+'\n')


if __name__ == '__main__':
    main()
