"""Replay selected adapters, then evaluate frozen prior populations without fitting."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import time

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.fit_scope_entailment_v9 import metrics
from training.gemma4_prompt_v9 import classifier_tokens
from training.gemma4_shared_prefix_v9 import capture_with_prefix
from training.gemma4_shared_tail_v9 import capture, replay
from training.gemma4_structured_v9 import portable_logits
from training.prepare_instruction_sft_v8 import target
from training.shared_tail_cache_v9 import to_mlx
from training.shared_tail_qualification_v9 import candidate_bindings, compare_replay, scores_from_logits
from training.shared_tail_training_v9 import make_classifier
from training.composite_tail_cache_v9 import CompositeReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    study, candidate, cache = [root/name for name in ('grounding-tail-prior-178', 'grounding-tail-fit-177', 'shared-tail-corpus-164')]
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen qualification source differs: ' + path)
    specs, _ = candidate_bindings(candidate)
    protocol = json.loads((study/'protocol.json').read_text())
    training = json.loads((candidate/'protocol.json').read_text())
    if protocol['development_batch_size'] != training['batch_size'] or protocol['batch_size'] != 1:
        raise ValueError('Qualification batch shapes must match training development and installed inference')
    manifest = json.loads((cache/'base-manifest.json').read_text())
    if args.base.name != manifest['revision']:
        raise ValueError('Wrong base revision')
    for name, record in manifest['files'].items():
        if digest(args.base/name) != record['sha256']:
            raise ValueError('Pinned base differs')
    software = json.loads((cache/'software.json').read_text())
    for name, expected in software['versions'].items():
        if version(name) != expected:
            raise ValueError('Dependency differs: ' + name)
    package = Path(distribution('mlx-lm').locate_file('mlx_lm'))
    for name, expected in software['implementation_sha256'].items():
        if digest(package/name) != expected:
            raise ValueError('MLX implementation differs')
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from safetensors.numpy import load_file
    mx.set_memory_limit(10*1024**3)
    mx.set_cache_limit(256*1024**2)
    reader = CompositeReader([Path(p) for p in training['caches']])
    entries = reader.entries
    prior = read_rows(study/'inputs.jsonl.gz')
    if len(prior) != protocol['rows']:
        raise ValueError('Prior population incomplete')
    base, tokenizer = load(str(args.base), tokenizer_config={'trust_remote_code': False})
    base.set_dtype(mx.float32)
    model = make_classifier(base.language_model,
        {lang: load_file(Path(head['path'])) for lang, head in training['heads'].items()},
        boundary=training['boundary'], lora=training['lora'])
    del base
    model.eval()
    labels = [tokenizer.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    if any(len(v) != 1 for v in labels):
        raise ValueError('Unexpected label tokenization')
    original = read_rows(root/'gemma4-readout-141/inputs.jsonl.gz')
    references = json.loads((cache/'protocol.json').read_text())['reference_indices']
    prefix = json.loads((cache/'common-prefix.json').read_text())
    output = study/'actual'
    output.mkdir(exist_ok=False)
    report = dict(complete=False, languages={}, source_freeze_sha256=digest(study/'source-freeze.json'),
        calibration_used_for_selection=False, fresh_test_inferred=False, synthetic_only=True,
        independently_annotated=False, installed_execution_verified=False, stable_release_ready=False)
    def encode(entry):
        tokens = classifier_tokens(tokenizer, entry['row'], [v[0] for v in labels])
        if len(tokens) != entry['token_count'] or hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest() != entry['token_sha256']:
            raise ValueError('Prompt token identity differs')
        return tokens
    def record(entry, value):
        row = entry['row']
        return dict(case_id=row['case_id'], language=row['language'], group=entry['group'],
            gold=entry.get('gold', int(row['review']) if 'review' in row else target(row)), score=float(value))
    for language in ('ro', 'uk'):
        started = time.monotonic()
        spec = specs[language]
        if spec['threshold'] != protocol['thresholds'][language]:
            raise ValueError('Qualification cutoff differs')
        checkpoint = candidate/'actual'/f'{language}-selected.safetensors'
        weights = mx.load(str(checkpoint))
        if set(weights) != set(dict(tree_flatten(model.trainable_parameters()))):
            raise ValueError('Incomplete or unexpected adapter/head weights')
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.trainable_parameters())
        del weights
        head = model.heads[language]
        tensors = {'0.weight': np.asarray(head.first.weight).copy(), '0.bias': np.asarray(head.first.bias).copy(),
            '3.weight': np.asarray(head.last.weight).copy(), '3.bias': np.asarray(head.last.bias).copy()}
        # Establish full-sequence and independent-head parity using training references only.
        feature_error, logit_error, reference_count = 0., 0., 0
        for index in references:
            entry = original[index]
            if entry['row']['language'] != language:
                continue
            if entry['partition'] != 'train':
                raise ValueError('Only training references may precede qualification')
            tokens = mx.array([encode(entry)])
            state = capture(model.base, tokens, start=training['boundary'])
            actual = replay(model.base, state)
            expected = model.base.model(tokens)[:, -1:, :]
            logits = model([state], [language])
            mx.eval(actual, expected, logits)
            if not bool(mx.all(mx.isfinite(actual)).item()) or not bool(mx.all(mx.isfinite(expected)).item()):
                raise ValueError('Nonfinite full-sequence reference')
            feature_error = max(feature_error, float((mx.max(mx.abs(actual-expected))/mx.maximum(1., mx.max(mx.abs(expected)))).item()))
            logit_error = max(logit_error, float(np.max(np.abs(np.asarray(logits)-portable_logits(np.asarray(expected[:, 0, :]), tensors)))))
            reference_count += 1
            del state, actual, expected, logits
            mx.clear_cache()
        if not reference_count or feature_error > protocol['full_forward_training_reference_relative_tolerance'] or logit_error > protocol['portable_head_logit_tolerance']:
            raise ValueError('Selected adapter full-sequence or portable-head parity differs')
        # Reproduce every development score before exposing any prior input to this model.
        indices = [i for i, r in enumerate(entries) if r['partition'] == 'dev' and r['row']['language'] == language]
        replay_rows = []
        for chosen in reader.development_batches(language, protocol['development_batch_size']):
            states = [to_mlx(s) for s in reader.read_many(chosen)]
            logits = model(states, [language]*len(chosen))
            mx.eval(logits)
            replay_rows += [record(entries[i], value) for i, value in zip(chosen, scores_from_logits(np.asarray(logits), spec))]
            del states, logits
            mx.clear_cache()
        parity = compare_replay(replay_rows, read_rows(candidate/'actual'/f'{language}-dev-predictions.jsonl'),
            spec['threshold'], protocol['development_score_tolerance'])
        parity.update(training_references=reference_count, full_forward_relative_error=feature_error, portable_head_logit_error=logit_error)
        (output/f'{language}-development-replay.json').write_text(json.dumps(parity, indent=2)+'\n')
        print('DEVELOPMENT_REPLAY', language, json.dumps(parity), flush=True)
        prefix_cache = None
        if prefix:
            prefix_cache = make_prompt_cache(model.base)
            mx.eval(model.base.model(mx.array([prefix]), cache=prefix_cache))
        selected = [r for r in prior if r['row']['language'] == language]
        predictions, all_logits = [], []
        cursor = 0
        while cursor < len(selected):
            stop = cursor+1
            while stop < len(selected) and stop-cursor < protocol['batch_size'] and selected[stop]['token_count'] == selected[cursor]['token_count']:
                stop += 1
            chosen = selected[cursor:stop]
            encoded = [encode(r) for r in chosen]
            if prefix and all(v[:len(prefix)] == prefix and len(v) > len(prefix) for v in encoded):
                state = capture_with_prefix(model.base, mx.array([v[len(prefix):] for v in encoded]), prefix_cache, start=training['boundary'])
            else:
                state = capture(model.base, mx.array(encoded), start=training['boundary'])
            features = replay(model.base, state)[:, 0, :]
            logits = model.heads[language](features)
            mx.eval(logits)
            values = np.asarray(logits).copy()
            all_logits.append(values)
            for entry, score in zip(chosen, scores_from_logits(values, spec)):
                row = record(entry, score)
                row['predicted'] = int(score >= spec['threshold'])
                predictions.append(row)
            del state, features, logits
            mx.clear_cache()
            cursor = stop
            if cursor % 100 < protocol['batch_size'] or cursor == len(selected):
                print('PRIOR', language, cursor, '/', len(selected), 'seconds', round(time.monotonic()-started, 2), flush=True)
        measured = metrics(predictions)
        (output/f'{language}-prior-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        np.savez(output/f'{language}-prior-logits.npz', logits=np.concatenate(all_logits))
        report['languages'][language] = dict(metrics=measured, prior_counts_pass=all(r['passes'] for r in measured.values()),
            development_replay=parity, cases=len(predictions), threshold=spec['threshold'], seconds=time.monotonic()-started)
        (output/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        print('QUALIFICATION', language, json.dumps(report['languages'][language]), flush=True)
    report.update(complete=True, prior_counts_pass=all(r['prior_counts_pass'] for r in report['languages'].values()),
        peak_mlx_bytes=mx.get_peak_memory(), finished_utc=datetime.now(timezone.utc).isoformat())
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
