"""Measure batch-shape replay drift on development only; no prior/test access."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.shared_tail_cache_v9 import to_mlx
from training.shared_tail_qualification_v9 import candidate_bindings, scores_from_logits
from training.shared_tail_training_v9 import make_classifier
from training.train_shared_tail_v9 import Reader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    candidate, cache = root/'shared-tail-fit-165', root/'shared-tail-corpus-164'
    specs, bindings = candidate_bindings(candidate)
    manifest = json.loads((cache/'base-manifest.json').read_text())
    if args.base.name != manifest['revision']:
        raise ValueError('Wrong base revision')
    for name, record in manifest['files'].items():
        if digest(args.base/name) != record['sha256']:
            raise ValueError('Pinned base differs')
    study = root/'shared-tail-replay-diagnostic-169'
    study.mkdir(exist_ok=False)
    bindings[str(Path(__file__).relative_to(Path.cwd()))] = digest(__file__)
    bindings[str(cache/'actual/report.json')] = digest(cache/'actual/report.json')
    (study/'source-freeze.json').write_text(json.dumps(bindings, indent=2)+'\n')
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from safetensors.numpy import load_file
    mx.set_memory_limit(10*1024**3)
    mx.set_cache_limit(256*1024**2)
    entries = read_rows(cache/'inputs.jsonl.gz')
    reader = Reader(cache, entries)
    protocol = json.loads((candidate/'protocol.json').read_text())
    base, _ = load(str(args.base), tokenizer_config={'trust_remote_code': False})
    base.set_dtype(mx.float32)
    model = make_classifier(base.language_model,
        {lang: load_file(Path(head['path'])) for lang, head in protocol['heads'].items()},
        boundary=protocol['boundary'], lora=protocol['lora'])
    del base
    model.eval()
    report = dict(complete=False, batches={}, calibration_inferred=False, fresh_test_inferred=False,
        development_only=True, weights_changed=False, source_freeze_sha256=digest(study/'source-freeze.json'))
    for language in ('ro', 'uk'):
        spec = specs[language]
        weights = mx.load(str(candidate/'actual'/f'{language}-selected.safetensors'))
        if set(weights) != set(dict(tree_flatten(model.trainable_parameters()))):
            raise ValueError('Unexpected tensor inventory')
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.trainable_parameters())
        del weights
        indices = [i for i, r in enumerate(entries) if r['partition'] == 'dev' and r['row']['language'] == language]
        expected = {r['case_id']: r for r in read_rows(candidate/'actual'/f'{language}-dev-predictions.jsonl')}
        for size in (4, 8, 1):
            started, rows = time.monotonic(), []
            for offset in range(0, len(indices), size):
                chosen = indices[offset:offset+size]
                states = [to_mlx(s) for s in reader.read_many(chosen)]
                logits = model(states, [language]*len(chosen))
                mx.eval(logits)
                for index, score in zip(chosen, scores_from_logits(np.asarray(logits), spec)):
                    reference = expected[entries[index]['row']['case_id']]
                    rows.append(dict(reference, score=float(score), predicted=int(score >= spec['threshold']),
                        reference_score=reference['score'], reference_predicted=reference['predicted']))
                del states, logits
                mx.clear_cache()
            errors = np.asarray([abs(r['score']-r['reference_score']) for r in rows])
            changes = [r for r in rows if r['predicted'] != r['reference_predicted']]
            result = dict(cases=len(rows), decision_changes=len(changes), score_max_error=float(errors.max()),
                score_quantiles=np.quantile(errors, [.5, .9, .99]).tolist(),
                worst_rows=sorted(rows, key=lambda r: -abs(r['score']-r['reference_score']))[:12],
                changed_rows=changes, seconds=time.monotonic()-started)
            report['batches'][language+'_'+str(size)] = result
            (study/f'{language}-batch-{size}-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (study/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
            print('REPLAY', language, size, json.dumps({k:v for k,v in result.items() if k not in ('worst_rows', 'changed_rows')}), flush=True)
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    (study/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
