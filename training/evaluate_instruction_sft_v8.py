"""Score a local Qwen instruction model without generating or executing text."""
import argparse
from copy import deepcopy
import hashlib
import json
import time
from pathlib import Path
from training.prepare_instruction_sft_v8 import messages, target


def completion_prefix(tokenizer, prompt_messages, label_ids):
    """Find the actual supervised label position, independent of the gold label.

    Some templates inject fixed assistant scaffolding even for empty reasoning.
    A generation header alone is then not the prefix preceding the class token.
    """
    options = [tokenizer.apply_chat_template(prompt_messages+[
        {'role': 'assistant', 'content': str(label)}], tokenize=True, return_dict=False)
        for label in (0, 1)]
    if len(options[0]) != len(options[1]):
        raise ValueError('Labels do not share a token-aligned template')
    changed = [i for i, (a, b) in enumerate(zip(*options)) if a != b]
    if len(changed) != 1:
        raise ValueError('Template must differ at exactly one label token')
    index = changed[0]
    if [tokens[index] for tokens in options] != label_ids:
        raise ValueError('Template label token differs from standalone label')
    return options[0][:index]


def classifier_tokens(tokenizer, row, label_ids, policy=None):
    prompt = messages(row) if policy is None else messages(row, policy=policy)
    return completion_prefix(tokenizer, prompt, label_ids)


def cached_batches(model, tokenized, ids, common, prefix_cache, batch_size):
    """Right padding is after each scored position and cannot affect its past."""
    import mlx.core as mx
    for begin in range(0, len(tokenized), batch_size):
        suffixes=[tokens[common:] for tokens in tokenized[begin:begin+batch_size]]
        width=max(map(len,suffixes)); count=len(suffixes)
        padded=[tokens+[0]*(width-len(tokens)) for tokens in suffixes]
        cache=deepcopy(prefix_cache)
        if common and count>1:
            for layer in cache:
                layer.state=tuple(mx.repeat(value,count,axis=0) for value in layer.state)
        tick=time.monotonic()
        hidden=model.model(mx.array(padded),cache=cache)
        hidden=hidden[mx.arange(count),mx.array([len(tokens)-1 for tokens in suffixes]),:][:,None,:]
        logits=(model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings else model.lm_head(hidden))[:,0,:].astype(mx.float32)
        pairs=mx.softmax(logits[:,mx.array(ids)],axis=-1);tops=mx.argmax(logits,axis=-1)
        mx.eval(pairs,tops);elapsed=time.monotonic()-tick
        for j in range(count):yield begin+j,float(pairs[j,1].item()),int(tops[j].item()),elapsed/count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--policy-file', type=Path)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.batch_size <= 8: p.error('batch-size must be between 1 and 8')
    a.output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(a.base, adapter_path=str(a.adapter) if a.adapter else None)
    model.set_dtype(mx.float32)  # Keep packed integer weights; stabilize cache/full arithmetic.
    ids = [tok.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    if any(len(v) != 1 for v in ids):
        raise ValueError('Labels must be single tokens')
    ids = [v[0] for v in ids]
    rows = [json.loads(line) for line in a.data.read_text(encoding='utf-8').splitlines()]
    policy = a.policy_file.read_text(encoding='utf-8') if a.policy_file else None
    tokenized = [classifier_tokens(tok, r, ids, policy) for r in rows]
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
    (a.output/'expected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected), encoding='utf-8')
    started = time.monotonic()
    with (a.output/'predictions.jsonl').open('w') as out:
        for i, score, top, seconds in cached_batches(model, tokenized, ids, common, prefix_cache, a.batch_size):
            row=rows[i]; tokens=tokenized[i]
            if i < max(2,a.batch_size):
                full = model(mx.array([tokens]))[0, -1].astype(mx.float32)
                full_pair = mx.softmax(full[mx.array(ids)])
                mx.eval(full_pair)
                error = abs(score-float(full_pair[1].item()))
                cache_check_max_error = max(cache_check_max_error, error)
                if error > .001 or int(mx.argmax(full).item()) != top:
                    raise ValueError(f'Cached inference differs: probability error={error}; cached top={top}; full top={int(mx.argmax(full).item())}')
            valid = top in ids
            predicted = int(score >= .5) if valid else None
            gold = target(row)
            result = dict(case_id=row['case_id'], language=row['language'], source=row.get('source'),
                gold=gold, predicted=predicted, score=score, valid_label_token=valid, seconds=seconds)
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
        base=Path(a.base).parent.parent.name, base_revision=Path(a.base).name, adapter=str(a.adapter) if a.adapter else None,
        data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        threshold=.5, invalid_outputs_counted_as_errors=True, cached_prefix_tokens=common, cache_check_max_error=cache_check_max_error,
        scoring_format='supervised_label_position_v2', input_normalization='NFC', floating_compute_dtype='float32', batch_size=a.batch_size, row_timing='amortized batch inference, excluding verification', policy_sha256=hashlib.sha256((policy or messages(rows[0])[0]['content']).encode()).hexdigest())
    if a.adapter:
        result['adapter_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.adapter.glob('*') if p.is_file()}
    from roguard.review_evaluation import evaluate
    result['strict_evaluation'] = evaluate(expected, [json.loads(l) for l in (a.output/'predictions.jsonl').read_text(encoding='utf-8').splitlines()])
    (a.output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('FINISHED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
