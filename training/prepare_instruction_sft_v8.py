"""Prepare an auditable local instruction-tuning pilot from training-only inputs."""
import argparse
import gzip
import hashlib
import json
import random
import unicodedata
from pathlib import Path
from training.probe_instruction_v8 import POLICY_V2
from training.review_data_v8 import input_key
from roguard.evidence import disposition


def target(row):
    return int(disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE')


def messages(row, answer=False, policy=POLICY_V2):
    turns = [dict(turn, text=unicodedata.normalize('NFC', turn['text'])) for turn in row['turns'][:row['focus']+1]]
    payload = json.dumps({'prior_turns': turns[:row['focus']], 'focal_message': turns[row['focus']]}, ensure_ascii=False)
    result = [{'role': 'system', 'content': policy}, {'role': 'user', 'content': payload}]
    if answer:
        result.append({'role': 'assistant', 'content': str(target(row))})
    return result


def read(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]


def sample(rows, per_group, seed):
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        kind = 'conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal'
        groups.setdefault((row['language'], kind, target(row)), []).append(row)
    if len(groups) != 8:
        raise ValueError('Missing language/kind/class group')
    chosen = []
    for key, group in sorted(groups.items()):
        rng.shuffle(group)
        if len(group) < per_group:
            raise ValueError('Insufficient examples: '+str(key))
        chosen.extend(group[:per_group])
    rng.shuffle(chosen)
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', required=True)
    a = p.parse_args()
    root = Path('research/v8-stabilization')
    train = []
    seen = {}
    for directory, lang in [('review-09-ro', 'ro'), ('review-08-uk', 'uk')]:
        for row in read(root/directory/'train.jsonl.gz'):
            if row['language'] != lang:
                continue
            key = input_key(row)
            if key in seen and seen[key] != target(row):
                raise ValueError('Conflicting training labels')
            if key not in seen:
                train.append(row)
                seen[key] = target(row)
    dev = read(root/'review-09-ro'/'dev.jsonl.gz')
    cal = read(root/'review-09-ro'/'cal.jsonl.gz')
    if set(map(input_key, dev+cal)) & set(seen):
        raise ValueError('Training overlaps evaluation')
    chosen = sample(train, 300, 8101)
    screen = sample(dev, 50, 8102)
    valid = sample(screen, 20, 8103)
    # Audit explicitly available concern-family labels. Missing labels are not
    # evidence of family independence; this pilot is an existing-regression study.
    tf = {r['family'] for r in chosen if r.get('family')}
    df = {r['family'] for r in dev if r.get('family')}
    if tf & df:
        raise ValueError('Annotated concern-family overlap')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
    a.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    lengths = []
    for name, rows, chat in [('train', chosen, True), ('valid', valid, True), ('screen', screen, False)]:
        raw = ''.join(json.dumps({'messages': messages(r, True)} if chat else r, ensure_ascii=False)+'\n' for r in rows)
        path = a.output/(name+'.jsonl'); path.write_text(raw, encoding='utf-8')
        hashes[path.name] = hashlib.sha256(raw.encode()).hexdigest()
        if chat:
            lengths.extend(len(tok.apply_chat_template(messages(r, True), tokenize=True, return_dict=False)) for r in rows)
    (a.output/'training-inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in chosen), encoding='utf-8')
    if max(lengths) > 1024:
        raise ValueError('Input would be truncated')
    base_path = Path(a.base)
    model_id = base_path.parent.parent.name.removeprefix('models--').replace('--', '/', 1)
    protocol = dict(model=model_id, model_revision=base_path.name, seed=8104, train_rows=len(chosen), valid_rows=len(valid), screen_rows=len(screen),
        policy=POLICY_V2, sha256=hashes, max_tokens=max(lengths), synthetic_only=True,
        annotated_family_overlap=sorted(tf & df), missing_family_labels=sum(not r.get('family') for r in chosen),
        family_independence_claim=False, calibration_status='previously inspected regression, never gradient training',
        fresh_48_case_probe_used=False, selection='400-case development screen; then full development and calibration; final test only after gates',
        thresholds='fixed 0.5 on normalized label-token scores for pilot; invalid top token counted as error',
        promotion='>=99% recall and <=5% false review per language and kind; fresh and runtime checks also required',
        pilot_steps=200, continuation='May extend training if development improves; record every examined checkpoint')
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    config = dict(model=model_id, train=True, data=str(a.output), seed=8104, fine_tune_type='lora', num_layers=8,
        batch_size=2, iters=200, val_batches=8, learning_rate=0.00005, steps_per_report=10,
        steps_per_eval=100, adapter_path=str(a.output/'adapters'), save_every=100,
        max_seq_length=1024, grad_checkpoint=True, grad_accumulation_steps=2,
        lora_parameters=dict(rank=8, dropout=0.0, scale=16.0), mask_prompt=True)
    # JSON is a valid YAML subset and avoids a new preparation dependency.
    (a.output/'config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(protocol, indent=2))


if __name__ == '__main__':
    main()
