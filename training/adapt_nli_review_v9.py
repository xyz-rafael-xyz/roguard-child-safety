"""Transfer a pinned multilingual entailment encoder to focal-message review.

This is an explicit research artifact. Calibration is inspected regression
evidence; no fresh probe is read and no runtime model is replaced.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

HYPOTHESIS = 'There is an emotional or safety concern about a real person, or an earlier concern is being withdrawn.'


def premise(row):
    focus = row['focus']
    return json.dumps({'previous_messages': row['turns'][:focus], 'focal_message': row['turns'][focus]}, ensure_ascii=False)


def read(path):
    return [json.loads(l) for l in gzip.decompress(Path(path).read_bytes()).decode().splitlines()]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='mps'); p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=2e-5); a = p.parse_args()
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from roguard.nli_v21 import verify_base, MODEL_ID, REVISION, BASE_HASHES
    from roguard.native_assets import reject_native_overrides, PADDING_MULTIPLE
    from training.prepare_instruction_sft_v8 import target
    from training.review_data_v8 import input_key
    from training.natural_contrasts_v9 import generate as natural
    from training.temporal_contrasts_v9 import generate as temporal
    from training.adapt_attention_v9 import thresholds_for
    from training.fit_review_v8 import group_metrics
    verify_base(Path(a.base)); reject_native_overrides(a.base, BASE_HASHES)
    torch.manual_seed(918); torch.set_num_threads(4); rng = random.Random(918)
    root = Path('research/v8-stabilization'); pool = []
    for name, lang in [('review-09-ro', 'ro'), ('review-08-uk', 'uk')]:
        pool += [r for r in read(root/name/'training-history.jsonl.gz') if r['language'] == lang]
    train = []; seen = set()
    for lang in ('ro', 'uk'):
        for native in (False, True):
            for label in (0, 1):
                candidates = [r for r in pool if r['language'] == lang and (r.get('source') == 'v7_training_frame_projection') == native and target(r) == label]
                rng.shuffle(candidates); count = 0
                for row in candidates:
                    key = input_key(row)
                    if key in seen: continue
                    train.append(row); seen.add(key); count += 1
                    if count == 500: break
                if count != 500: raise ValueError('Missing balanced training group')
    for row in natural()+temporal():
        if input_key(row) not in seen: train.append(row); seen.add(input_key(row))
    rows = {'train': train, 'dev': read(root/'review-09-ro/dev.jsonl.gz'), 'cal': read(root/'review-09-ro/cal.jsonl.gz')}
    if seen & {input_key(r) for r in rows['dev']+rows['cal']}: raise ValueError('Evaluation overlap')
    a.output.mkdir(exist_ok=False)
    for split, values in rows.items():
        (a.output/(split+'.jsonl.gz')).write_bytes(gzip.compress((''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values)).encode(), mtime=0))
    (a.output/'training-history.jsonl.gz').write_bytes((a.output/'train.jsonl.gz').read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(a.base, local_files_only=True, trust_remote_code=False)
    if tokenizer.padding_side != 'right': raise ValueError('NLI transfer requires right padding')
    base = AutoModelForSequenceClassification.from_pretrained(a.base, local_files_only=True, trust_remote_code=False, use_safetensors=True).float()
    if base.config.id2label.get(0) != 'entailment' or base.config.id2label.get(2) != 'contradiction': raise ValueError('Unexpected NLI labels')
    architecture = dict(r=8, lora_alpha=16, lora_dropout=.05, target_modules=['query_proj', 'value_proj'], modules_to_save=['classifier', 'pooler'])
    model = get_peft_model(base, LoraConfig(**architecture)).to(a.device)
    tokens = {}; truth = {}
    for split, values in rows.items():
        tokens[split] = tokenizer([premise(r) for r in values], [HYPOTHESIS]*len(values), truncation=False, padding=False)
        if max(map(len, tokens[split]['input_ids'])) > 512: raise ValueError('Input would exceed 512 tokens')
        truth[split] = np.array([target(r) for r in values])
    def batch(split, ix):
        return {k:v.to(a.device) for k,v in tokenizer.pad([{k:v[i] for k,v in tokens[split].items()} for i in ix], return_tensors='pt', pad_to_multiple_of=PADDING_MULTIPLE).items()}
    def score(split):
        model.eval(); values = []
        with torch.inference_mode():
            for offset in range(0, len(rows[split]), 8):
                logits = model(**batch(split, list(range(offset, min(offset+8, len(rows[split])))))).logits.float()
                values.extend(torch.sigmoid(logits[:, 0]-logits[:, 2]).cpu().tolist())
                if a.device == 'mps' and (offset//8+1)%25 == 0: torch.mps.empty_cache()
        if not np.isfinite(values).all(): raise ValueError('Nonfinite NLI inference score')
        return np.array(values)
    def save_scores(split, scores, thresholds):
        records = [dict(case_id=r['case_id'], language=r['language'], gold=int(y), score=float(s), predicted=int(s >= thresholds[r['language']])) for r,y,s in zip(rows[split], truth[split], scores)]
        (a.output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in records))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.learning_rate)
    record = dict(synthetic_only=True, selection_partition='development', base=MODEL_ID, revision=REVISION,
                  training_rows=len(train), hypothesis=HYPOTHESIS, learning_rate=a.learning_rate, epochs=[],
                  padding_multiple=PADDING_MULTIPLE, mps_cache_release_interval=25)
    best = None
    for epoch in range(1, a.epochs+1):
        model.train(); order = list(range(len(train))); rng.shuffle(order); losses = []; start = time.monotonic()
        for offset in range(0, len(order), 8):
            ix = order[offset:offset+8]; optimizer.zero_grad()
            logits = model(**batch('train', ix)).logits.float()
            labels = torch.tensor([0 if truth['train'][i] else 2 for i in ix], device=a.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=.02)
            if not torch.isfinite(loss): raise ValueError('Nonfinite NLI training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True); optimizer.step(); losses.append(float(loss.detach().cpu()))
            if a.device == 'mps' and (offset//8+1)%25 == 0: torch.mps.empty_cache()
            if offset % 800 == 0: print('epoch', epoch, 'row', offset, 'of', len(train), 'loss', losses[-1], flush=True)
        scores = score('dev'); thresholds = thresholds_for(rows['dev'], truth['dev'], scores)
        metrics = group_metrics(rows['dev'], truth['dev'], scores, thresholds)
        entry = dict(epoch=epoch, loss=sum(losses)/len(losses), seconds=time.monotonic()-start, thresholds=thresholds, development=metrics)
        record['epochs'].append(entry); print(json.dumps(entry), flush=True)
        rank = (max(v['false_review_rate'] for v in metrics.values()), sum(v['false_review_rate'] for v in metrics.values()))
        artifact = dict(format='roguard-nli-review-experiment-v1', base=MODEL_ID, revision=REVISION, hypothesis=HYPOTHESIS,
                        architecture=architecture, thresholds=thresholds, padding_multiple=PADDING_MULTIPLE,
                        tensors={k:v.detach().cpu().tolist() for k,v in get_peft_model_state_dict(model).items()})
        raw = gzip.compress(json.dumps(artifact, separators=(',', ':')).encode(), mtime=0)
        (a.output/f'epoch-{epoch}-review.json.gz').write_bytes(raw)
        if best is None or rank < best:
            best = rank; record['selected_epoch'] = epoch; chosen_thresholds = thresholds
            (a.output/'review.json.gz').write_bytes(raw); save_scores('dev', scores, thresholds)
        (a.output/'development.json').write_text(json.dumps(record, indent=2)+'\n')
    selected = json.loads(gzip.decompress((a.output/'review.json.gz').read_bytes()))
    set_peft_model_state_dict(model, {k:torch.tensor(v, dtype=torch.float32) for k,v in selected['tensors'].items()})
    scores = score('cal'); save_scores('cal', scores, chosen_thresholds)
    record['calibration_audit'] = group_metrics(rows['cal'], truth['cal'], scores, chosen_thresholds)
    record['artifact_sha256'] = hashlib.sha256((a.output/'review.json.gz').read_bytes()).hexdigest()
    (a.output/'development.json').write_text(json.dumps(record, indent=2)+'\n')
    print('FINISHED', json.dumps(record['calibration_audit']), flush=True)


if __name__ == '__main__': main()
