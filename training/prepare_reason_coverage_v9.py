"""Expand training coverage by policy reason and semantic attributes, not label alone."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random

from training.policy_entailment_v9 import targets, HYPOTHESES
from training.revision_contrasts_v9 import generate
from training.train_paired_nli_v9 import premise
from training.train_bare_distress_v9 import digest, read

STUDY = Path('research/v9-stabilization/reason-coverage-123')
PARENT = Path('research/v9-stabilization/policy-entailment-121/candidate')
SPLITS = ('ro_dev', 'uk_dev', 'ro_scope_dev', 'uk_scope_dev', 'uk_stress',
          'ro_cal', 'uk_cal', 'ro_exposed', 'uk_exposed')


def stratum(row):
    a = row['attributes']
    return (row['language'], row['source'], *targets(row), a['subject'], a['assertion'], a['time'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from roguard.nli_v21 import verify_base
    verify_base(args.base)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    output = STUDY / 'data'
    output.mkdir(exist_ok=False)
    hashes = {}

    def tracked(path):
        hashes[str(path)] = digest(path)
        return read(path)

    def keys(rows):
        result = []
        for start in range(0, len(rows), 128):
            batch = rows[start:start+128]
            encoded = tokenizer([premise(r) for r in batch for _ in HYPOTHESES],
                                list(HYPOTHESES)*len(batch), padding=False, truncation=False)
            if any(len(v) > 512 for v in encoded['input_ids']):
                raise ValueError('No silent truncation')
            for i in range(len(batch)):
                pair = encoded['input_ids'][2*i:2*i+2]
                result.append(hashlib.sha256(json.dumps(pair, separators=(',', ':')).encode()).hexdigest())
        return result

    partitions = {s: tracked(PARENT / (s+'.jsonl.gz')) for s in SPLITS}
    fresh = []
    for lang in ('ro', 'uk'):
        path = Path('research/v9-stabilization/paired-scope-117/data') / (lang+'-test.jsonl')
        hashes[str(path)] = digest(path)
        fresh += [json.loads(line) for line in path.read_text().splitlines()]
    protected = set(keys(fresh + [r for values in partitions.values() for r in values]))
    train = tracked(PARENT / 'train.jsonl.gz')
    original_count = len(train)
    selected = {}
    for row, key in zip(train, keys(train)):
        if key in protected or key in selected:
            raise ValueError('Parent training overlap')
        selected[key] = targets(row)
    authored = [r for lang in ('ro', 'uk') for r in generate(lang)]
    for row, key in zip(authored, keys(authored)):
        if key in protected or key in selected:
            raise ValueError('Authored training overlap')
        selected[key] = targets(row)
        train.append(row)
    pool = {}
    for language, name in (('ro', 'native-full-22-ro'), ('uk', 'bare-distress-95/uk')):
        history = tracked(Path('research/v9-stabilization') / name / 'training-history.jsonl.gz')
        for row, key in zip(history, keys(history)):
            if row['language'] != language:
                raise ValueError('History language differs')
            if key in protected:
                continue
            existing = selected.get(key, targets(pool[key]) if key in pool else None)
            if existing is not None and existing != targets(row):
                raise ValueError('Contradictory policy factors for equal token inputs')
            if key not in selected:
                pool[key] = row
    grouped = defaultdict(list)
    for key, row in pool.items():
        grouped[stratum(row)].append((key, row))
    rng = random.Random(1123)
    current = Counter(stratum(r) for r in train)
    additions = Counter()
    for group, candidates in sorted(grouped.items()):
        rng.shuffle(candidates)
        # Include every eligible historical revision; fill other semantic strata
        # to 32 distinct examples, retaining the entire parent replay unchanged.
        count = len(candidates) if group[3] else max(0, 32-current[group])
        for key, row in candidates[:count]:
            train.append(row)
            selected[key] = targets(row)
            additions[row['language']] += 1
    if set(keys(train)) & protected:
        raise ValueError('Protected token overlap')
    partitions['train'] = train
    for split, rows in partitions.items():
        (output/(split+'.jsonl.gz')).write_bytes(gzip.compress(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode(), mtime=0))
    (output/'training-history.jsonl.gz').write_bytes((output/'train.jsonl.gz').read_bytes())
    summary = dict(synthetic_only=True, independently_annotated=False, training_rows=len(train),
                   inherited_rows=original_count, authored_revision_rows=len(authored),
                   historical_additions=dict(additions), seed=1123, minimum_stratum_quota=32,
                   protected_token_overlap=0, fresh_test_neurally_inferred=False,
                   source_input_sha256=hashes,
                   reason_counts=[dict(language=lang, support=s, revision=r, rows=n)
                                  for (lang, s, r), n in sorted(Counter((v['language'], *targets(v)) for v in train).items())])
    (output/'coverage.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
