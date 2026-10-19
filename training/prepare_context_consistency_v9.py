"""Prepare Ukrainian NLI continuation with matched-history supervision."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.policy_entailment_v9 import HYPOTHESES, targets
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
    study = root/'context-consistency-162'
    study.mkdir(exist_ok=False)
    output = study/'data'
    output.mkdir()
    inputs = {}
    def tracked(path):
        inputs[str(path)] = digest(path)
        return read_rows(path)
    def keys(rows):
        result = []
        for start in range(0, len(rows), 128):
            part = rows[start:start+128]
            ids = tokenizer([premise(r) for r in part for _ in HYPOTHESES], list(HYPOTHESES)*len(part), padding=False, truncation=False)['input_ids']
            if any(len(v) > 512 for v in ids):
                raise ValueError('No truncated training or evaluation')
            result.extend(hashlib.sha256(json.dumps(ids[i:i+2], separators=(',', ':')).encode()).hexdigest() for i in range(0, len(ids), 2))
        return result
    parent = root/'reason-coverage-123/candidate'
    history = tracked(parent/'training-history.jsonl.gz')
    splits = {s: tracked(parent/(s+'.jsonl.gz')) for s in ('uk_dev', 'uk_scope_dev', 'uk_stress', 'uk_cal', 'uk_exposed')}
    splits['uk_transition_dev'] = tracked(root/'context-transition-161/uk-dev.jsonl.gz')
    fresh = [r for lang in ('ro', 'uk') for r in tracked(root/'paired-scope-117/data'/f'{lang}-test.jsonl')]
    protected = set(keys(fresh+[r for rows in splits.values() for r in rows]))
    history_keys = keys(history)
    if set(history_keys) & protected:
        raise ValueError('Inherited training overlaps protected data, including new transition development')
    # Round-robin strata preserve rare policy reasons without assuming one
    # message label adequately represents the two NLI tasks.
    groups = defaultdict(list)
    for row in history:
        if row['language'] == 'uk':
            a = row['attributes']
            groups[(row['source'], *targets(row), a['subject'], a['assertion'], a['time'])].append(row)
    rng = random.Random(1162)
    for rows in groups.values():
        rng.shuffle(rows)
    replay = []
    while len(replay) < 4096 and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(replay) < 4096:
                replay.append(groups[key].pop())
    authored = []
    for source in ('indirect-distress-135', 'freeform-scope-136'):
        authored += tracked(root/source/'uk-train.jsonl.gz')
    authored += tracked(root/'context-transition-161/uk-train.jsonl.gz')
    train = replay+authored
    encoded = keys(train)
    if len(encoded) != len(set(encoded)) or set(encoded) & protected:
        raise ValueError('Training duplicate or protected overlap')
    splits['train'] = train
    combined = {key: row for key, row in zip(history_keys, history)}
    combined.update(zip(encoded, train))
    splits['training-history'] = list(combined.values())
    for name, rows in splits.items():
        (output/(name+'.jsonl.gz')).write_bytes(gzip.compress(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode(), mtime=0))
    protocol = dict(study=162, created_utc=datetime.now(timezone.utc).isoformat(), language='uk',
        parent='Frozen123 selected bilingual NLI checkpoint; this continuation is Ukrainian only',
        training_rows=len(train), replay_rows=len(replay), authored_rows=len(authored), inherited_history_rows=len(history),
        new_objective='Balanced reason crossentropy +0.2 paired margin +0.1 symmetric KL across identical focal messages with differing history',
        trainable='Last four encoder blocks, pooler and classifier', seed=1162, epochs=2, learning_rate=5e-6,
        selection='UK per-group99/5 two-reason frontier, including held-out161 transitions; checkpoint and cutoffs fixed before calibration',
        output='Research-only Ukrainian artifact; not qualified for Romanian or direct production loading',
        design_informed_by='158 and160 context-switch misses; no failed-case wording copied',
        protected_token_overlap=0, inherited_protected_token_overlap=0, fresh_test_neurally_inferred=False,
        source_input_sha256=inputs, synthetic_only=True, independently_annotated=False, stable_release_ready=False,
        training_reason_counts=[dict(support=s, revision=r, rows=n) for (s, r), n in sorted(Counter(tuple(targets(row)) for row in train).items())])
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    paths = list(output.glob('*'))+[study/'protocol.json', parent/'review.json', parent/'development.json',
        root/'context-transition-161/source-freeze.json']
    paths += [Path('training')/p for p in ('prepare_context_consistency_v9.py', 'train_context_consistency_v9.py',
        'context_consistency_v9.py', 'policy_entailment_v9.py', 'train_paired_nli_v9.py',
        'budgeted_entailment_frontier_v9.py', 'entailment_frontier_v9.py', 'verify_candidate_pipeline_v9.py',
        'train_bare_distress_v9.py', 'prepare_instruction_sft_v8.py')]
    paths += [Path('src/roguard')/p for p in ('semantic.py', 'nli_v21.py', 'native_assets.py')]
    (study/'source-freeze.json').write_text(json.dumps({str(p): digest(p) for p in paths}, indent=2)+'\n')
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    main()
