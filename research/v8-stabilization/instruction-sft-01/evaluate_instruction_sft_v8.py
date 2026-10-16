"""Score a local Qwen instruction model without generating or executing text."""
import argparse
from copy import deepcopy
import hashlib
import json
import time
from pathlib import Path
from training.prepare_instruction_sft_v8 import messages, target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(a.base, adapter_path=str(a.adapter) if a.adapter else None)
    ids = [tok.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    if any(len(v) != 1 for v in ids):
        raise ValueError('Labels must be single tokens')
    ids = [v[0] for v in ids]
    rows = [json.loads(line) for line in a.data.read_text().splitlines()]
    tokenized = [tok.apply_chat_template(messages(r), tokenize=True, add_generation_prompt=True, return_dict=False) for r in rows]
    if not tokenized or any(len(t) > 1024 for t in tokenized):
        raise ValueError('Empty evaluation or input exceeds context limit')
    common = 0
    for values in zip(*tokenized):
        if len(set(values)) != 1: break
        common += 1
    common = min(common, min(map(len, tokenized))-1)
    from mlx_lm.models.cache import make_prompt_cache
    prefix_cache = make_prompt_cache(model)
    if common:
        mx.eval(model.model(mx.array([tokenized[0][:common]]), cache=prefix_cache))
    cache_check_max_error = 0.0
    metrics = {}
    expected = [dict(case_id=r['case_id'], language=r['language'], gold=target(r),
        kind='conversational' if r.get('source') == 'v7_training_frame_projection' else 'formal',
        invariant_group=r.get('invariant_group'), pair_group=r.get('pair_group')) for r in rows]
    (a.output/'expected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected))
    started = time.monotonic()
    with (a.output/'predictions.jsonl').open('w') as out:
        for i, row in enumerate(rows):
            tokens = tokenized[i]
            if len(tokens) > 1024:
                raise ValueError('Input exceeds registered context limit; refusing truncation')
            start = time.monotonic()
            # Causal hidden state at the final prompt position; avoid materializing
            # vocabulary logits for every preceding position.
            hidden = model.model(mx.array([tokens[common:]]), cache=deepcopy(prefix_cache))[:, -1:, :]
            logits = model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings else model.lm_head(hidden)
            logits = logits[0, 0].astype(mx.float32)
            pair = mx.softmax(logits[mx.array(ids)])
            top = mx.argmax(logits)
            mx.eval(pair, top)
            score = float(pair[1].item())
            if i < 2:
                full = model(mx.array([tokens]))[0, -1].astype(mx.float32)
                full_pair = mx.softmax(full[mx.array(ids)])
                mx.eval(full_pair)
                error = abs(score-float(full_pair[1].item()))
                cache_check_max_error = max(cache_check_max_error, error)
                if error > .001 or int(mx.argmax(full).item()) != int(top.item()):
                    raise ValueError('Cached inference differs from full prompt')
            valid = int(top.item()) in ids
            predicted = int(score >= .5) if valid else None
            gold = target(row)
            result = dict(case_id=row['case_id'], language=row['language'], source=row.get('source'),
                gold=gold, predicted=predicted, score=score, valid_label_token=valid, seconds=time.monotonic()-start)
            out.write(json.dumps(result)+'\n'); out.flush()
            kind = 'conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal'
            key = row['language']+'_'+kind
            m = metrics.setdefault(key, dict(positive=0, negative=0, missed=0, false_reviews=0, invalid=0))
            m['positive' if gold else 'negative'] += 1
            m['missed'] += gold == 1 and predicted != 1
            m['false_reviews'] += gold == 0 and predicted != 0
            m['invalid'] += not valid
            if (i+1) % 25 == 0:
                print(i+1, json.dumps(metrics), flush=True)
    result = dict(metrics=metrics, seconds=time.monotonic()-started, synthetic_only=True,
        base=a.base, adapter=str(a.adapter) if a.adapter else None,
        data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        threshold=.5, invalid_outputs_counted_as_errors=True, cached_prefix_tokens=common, cache_check_max_error=cache_check_max_error)
    if a.adapter:
        result['adapter_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.adapter.glob('*') if p.is_file()}
    from roguard.review_evaluation import evaluate
    result['strict_evaluation'] = evaluate(expected, [json.loads(l) for l in (a.output/'predictions.jsonl').read_text().splitlines()])
    (a.output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print('FINISHED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
