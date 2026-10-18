"""Prepare training-only replay with explicit coverage of rare interpretation fields."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random

from training.natural_contrasts_v9 import generate as natural
from training.temporal_contrasts_v9 import generate as temporal
from training.prepare_instruction_sft_v8 import target
from training.review_data_v8 import input_key
from training.semantic_interpreter_v9 import FIELDS, aligned_tokens, answer, chat


def read(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path('research/v8-stabilization')
    rng = random.Random(920)
    selected = []
    seen = set()
    inputs = {}
    for language, directory in (('ro', 'review-09-ro'), ('uk', 'review-08-uk')):
        path = root/directory/'training-history.jsonl.gz'
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        history = [r for r in read(path) if r['language'] == language]
        additions = [r for r in natural()+temporal() if r['language'] == language]
        start = len(selected)
        def take(pool, count):
            candidates = list(pool)
            rng.shuffle(candidates)
            added = 0
            for row in candidates:
                key = input_key(row)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(row)
                added += 1
                if added == count:
                    return
            raise ValueError(f'Insufficient unique {language} examples: {added}/{count}')
        for label in (0, 1):
            take([r for r in history if r.get('source') == 'v7_training_frame_projection' and target(r) == label], 400)
        for assertion in ('NEGATED', 'UNCERTAIN', 'RETRACTED'):
            take([r for r in history if r['attributes']['assertion'] == assertion], 128)
        for time in ('PAST', 'HYPOTHETICAL'):
            take([r for r in history if r['attributes']['time'] == time], 128)
        for label in (0, 1):
            take([r for r in additions if target(r) == label], 200)
        take(history, 2048-(len(selected)-start))
    # Existing development/calibration inputs are excluded, not scored or used
    # to author rows. Reserved fresh inputs are neither opened nor tokenized.
    for directory in ('review-09-ro', 'review-08-uk'):
        for split in ('dev', 'cal'):
            path = root/directory/(split+'.jsonl.gz')
            if seen & {input_key(r) for r in read(path)}:
                raise ValueError('Evaluation overlap')
    valid_path = Path('research/v9-stabilization/interpreter-14/valid.jsonl')
    valid = [json.loads(line) for line in valid_path.read_text().splitlines()]
    if seen & {input_key(r) for r in valid}:
        raise ValueError('Validation overlap')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    lengths = [len(aligned_tokens(tokenizer, chat(r), answer(r))[0]) for r in selected+valid]
    if max(lengths) > 1024:
        raise ValueError('Overlength training input')
    rng.shuffle(selected)
    args.output.mkdir(exist_ok=False)
    for split, rows in (('train', selected), ('valid', valid)):
        (args.output/(split+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    config = json.loads(Path('research/v9-stabilization/interpreter-14/config-v3.json').read_text())
    config.update(seed=920, data=str(args.output), iters=len(selected),
                  adapter_path=str(args.output/'adapters'))
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    report = dict(synthetic_only=True, reserved_probes_used=False, stable_release_ready=False,
                  training_rows=len(selected), validation_rows=len(valid), max_tokens=max(lengths),
                  selection='Minimum full validation class-token loss; no calibration selection.',
                  comparison='Same pinned 4B base, prompt, rank and learning rate as interpreter-14; broader training coverage.',
                  counts={language: {f: dict(Counter(r['attributes'][f] for r in selected if r['language']==language))
                                     for f in FIELDS} for language in ('ro','uk')},
                  source_sha256=inputs,
                  file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()})
    (args.output/'protocol.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
