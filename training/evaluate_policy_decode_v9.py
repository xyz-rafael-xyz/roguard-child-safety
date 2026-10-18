"""Evaluate adaptive policy integration with the same structured local model."""
import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from roguard.evidence import ONTOLOGY
from roguard.review_evaluation import evaluate
from training.policy_decode_v9 import decode
from training.prepare_instruction_sft_v8 import target
from training.semantic_interpreter_v9 import FIELDS, aligned_tokens, chat


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-queries', type=int, default=32)
    a = p.parse_args()
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    mx.set_cache_limit(512*1024*1024)
    model, tok = load(a.base, adapter_path=str(a.adapter) if a.adapter else None)
    model.set_dtype(mx.float32)
    digit_tokens = [tok.encode(str(i), add_special_tokens=False) for i in range(4)]
    if any(len(v) != 1 for v in digit_tokens):
        raise ValueError('Labels must be single tokens')
    ids = [v[0] for v in digit_tokens]
    rows = [json.loads(l) for l in a.data.read_text().splitlines()]
    prepared = [aligned_tokens(tok, chat(row), {f: 0 for f in FIELDS}) for row in rows]
    if not rows or max(len(t) for t, _ in prepared) > 1024:
        raise ValueError('Empty or overlength dataset')
    common = 0
    for values in zip(*(t[:p[0]] for t, p in prepared)):
        if len(set(values)) != 1: break
        common += 1
    common = min(common, min(p[0] for _, p in prepared)-1)
    common_cache = make_prompt_cache(model)
    if common:
        mx.eval(model(mx.array([prepared[0][0][:common]]), cache=common_cache))
    expected = [dict(case_id=r['case_id'], language=r['language'], gold=target(r),
                     kind='conversational' if r.get('source') == 'v7_training_frame_projection' else 'formal') for r in rows]
    a.output.mkdir(exist_ok=False)
    predictions = []
    start = time.monotonic()
    max_error = 0.0
    with (a.output/'predictions.jsonl').open('w') as output:
        for i, (row, (template, positions)) in enumerate(zip(rows, prepared)):
            cache_pool = OrderedDict()
            def conditional(prefix, field):
                nonlocal max_error
                tokens = list(template)
                for j, label in enumerate(prefix):
                    tokens[positions[j]] = ids[ONTOLOGY[FIELDS[j]].index(label)]
                position = positions[len(prefix)]
                parent = prefix[:-1]
                if prefix and parent in cache_pool:
                    previous, offset = cache_pool[parent]
                    cache = deepcopy(previous)
                    cache_pool.move_to_end(parent)
                else:
                    cache, offset = deepcopy(common_cache), common
                logits = model(mx.array([tokens[offset:position]]), cache=cache)[0, -1].astype(mx.float32)
                allowed = ids[:len(ONTOLOGY[field])]
                probabilities = mx.softmax(logits[mx.array(allowed)])
                top = mx.argmax(logits)
                mx.eval(probabilities, top)
                if i == 0:
                    full = model(mx.array([tokens[:position]]))[0, -1].astype(mx.float32)
                    check = mx.softmax(full[mx.array(allowed)])
                    mx.eval(check)
                    error = float(mx.max(mx.abs(check-probabilities)).item())
                    max_error = max(error, max_error)
                    if error > .001 or int(mx.argmax(full).item()) != int(top.item()):
                        raise ValueError('Cached/uncached tree scoring differs')
                cache_pool[prefix] = (cache, position)
                # Bound KV memory independently of the number of tree branches.
                while len(cache_pool) > 2: cache_pool.popitem(last=False)
                if int(top.item()) not in allowed:
                    return None
                return [float(v) for v in probabilities.tolist()]
            decision = decode(conditional, max_queries=a.max_queries)
            result = dict(case_id=row['case_id'], language=row['language'], gold=target(row), **decision)
            predictions.append(result)
            output.write(json.dumps(result)+'\n'); output.flush()
            cache_pool.clear()
            if (i+1) % 25 == 0:
                print(i+1, 'correct', sum(p['predicted'] == p['gold'] for p in predictions),
                      'unresolved', sum(p['predicted'] is None for p in predictions), flush=True)
    record = dict(synthetic_only=True, stable_release_ready=False, reserved_probes_used=False,
                  base_revision=Path(a.base).name, max_queries=a.max_queries, threshold=.5,
                  adapter_sha256=hashlib.sha256((a.adapter/'adapters.safetensors').read_bytes()).hexdigest() if a.adapter else None,
                  data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(), seconds=time.monotonic()-start,
                  strict_evaluation=evaluate(expected, predictions), cache_check_max_error=max_error,
                  mean_queries=sum(r['queries'] for r in predictions)/len(predictions),
                  probabilities_are_calibrated=False)
    (a.output/'expected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected))
    (a.output/'metrics.json').write_text(json.dumps(record, indent=2)+'\n')
    print('FINISHED', json.dumps(record), flush=True)


if __name__ == '__main__': main()
