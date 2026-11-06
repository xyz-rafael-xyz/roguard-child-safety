"""Measure fixed committee cost against v9 and sequential component passes."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import statistics
import time

from roguard.committee_bundle import load_committee
from roguard.gemma4_committee import COMPONENTS
from roguard.gemma4_contract import classifier_tokens
from roguard.gemma4_review import _LOCK, _load_weights
from roguard.trajectory_data import file_sha, input_identity, prefix, read_jsonl
from roguard.trajectory_evaluation import checkpoint_id
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'base', 'selector', 'runtime_study', 'study'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--sha256', required=True); a = p.parse_args()
    if a.study.exists():
        raise ValueError('Refusing to overwrite performance comparison')
    previous = json.loads((a.runtime_study/'report.json').read_text())
    binding = json.loads((a.runtime_study/'protocol.json').read_text())
    cases_path = Path('research/v10-evidence/broad-v9-regression/cases.jsonl')
    if previous['decision'] != 'BINARY_RUNTIME_ENGINEERING_PASS_ONLY' or file_sha(cases_path) != binding['inputs']['broad']:
        raise ValueError('Expected the fixed passing runtime and input population')
    selector = TrajectoryModel(a.selector)
    if selector.identity != binding['selector_sha256']:
        raise ValueError('Memory selector differs')
    cases = read_jsonl(cases_path); a.study.mkdir(parents=True)
    report = {'complete': False, 'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-36.json'),
        'runtime_report_sha256': file_sha(a.runtime_study/'report.json'), 'bundle_sha256': a.sha256,
        'environment': {'machine': platform.machine(), 'system': platform.system(),
                        'python': platform.python_version(), 'mlx': version('mlx'), 'mlx_lm': version('mlx-lm')},
        'languages': {}, 'independent_accuracy_test': False, 'per_request_benchmark': False, 'stable_release_ready': False}
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    for lang in ('ro', 'uk'):
        candidates = []
        for case in (c for c in cases if c['language'] == lang):
            for point in case['checkpoints']:
                view, inference = prepare_memory(prefix(case, point['focus']), selector, 1200, preserve_complete=True)
                if view['mandatory_overflow']:
                    raise ValueError('Previously supported memory view now overflows')
                candidates.append((sum(len(t['text']) for t in inference['turns']), checkpoint_id(case, point), inference))
        candidates.sort(key=lambda r: (r[0], r[1]))
        chosen = [candidates[round(i*(len(candidates)-1)/11)] for i in range(12)]
        if len({r[1] for r in chosen}) != 12:
            raise ValueError('Incomplete benchmark length grid')
        payloads = [r[2] for r in chosen]
        start = time.perf_counter(); model = load_committee(a.bundle, expected_sha256=a.sha256, base=a.base, language=lang)
        if model.identity != binding['committee_identities'][lang]:
            raise ValueError('Benchmark committee identity differs')
        model.predict(payloads[0]); startup = time.perf_counter()-start
        shared = model._engine; engine = shared.engine; mx = engine.mx; specs = model._specs
        with _LOCK:
            reference = shared.scores(payloads, specs)
            def run(kind, batch):
                try:
                    if kind == 'shared':
                        return shared.scores(batch, specs)
                    names = ('v9',) if kind == 'v9' else COMPONENTS
                    by_model = {}
                    for name in names:
                        _load_weights(engine.model, shared.tensors[name])
                        by_model[name] = engine.scores(batch, specs[name])
                    return [{name: {'score': values[i]} for name, values in by_model.items()} for i in range(len(batch))]
                finally:
                    _load_weights(engine.model, shared.tensors['v9'])
            for kind in ('v9', 'sequential', 'shared'):
                run(kind, payloads[:2])
            timings = []; rng = random.Random(1038)
            for repetition in range(3):
                kinds = ['v9', 'sequential', 'shared']; rng.shuffle(kinds)
                for kind in kinds:
                    mx.clear_cache()
                    if hasattr(mx, 'reset_peak_memory'):
                        mx.reset_peak_memory()
                    start = time.perf_counter(); actual = run(kind, payloads); elapsed = time.perf_counter()-start
                    maximum = 0.
                    for expected, observed in zip(reference, actual):
                        for name, output in observed.items():
                            maximum = max(maximum, abs(output['score']-expected[name]['score']))
                    if len(actual) != 12 or maximum > 1e-7:
                        raise ValueError('Benchmark comparator outputs differ')
                    row = {'repetition': repetition, 'kind': kind, 'seconds': elapsed,
                           'mlx_peak_bytes': mx.get_peak_memory() if hasattr(mx, 'get_peak_memory') else None,
                           'maximum_component_score_error': maximum}
                    timings.append(row); print(json.dumps({'language': lang, **row}), flush=True)
        medians = {kind: statistics.median(r['seconds'] for r in timings if r['kind'] == kind) for kind in ('v9', 'sequential', 'shared')}
        report['languages'][lang] = {'startup_including_first_prediction_seconds': startup,
            'inputs': [{'checkpoint_id': cid, 'characters': count, 'tokens': len(classifier_tokens(engine.tokenizer, payload)),
                        'inference_input_digest': input_identity(payload)} for count, cid, payload in chosen],
            'warm_batch_size': 12, 'measurements': timings, 'median_batch_seconds': medians,
            'shared_over_v9_time_ratio': medians['shared']/medians['v9'],
            'shared_over_sequential_time_ratio': medians['shared']/medians['sequential']}
        (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        del run, reference, actual, model, shared, engine
        import gc
        gc.collect(); mx.clear_cache()
    report['complete'] = True
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: {n: v[n] for n in ('median_batch_seconds', 'shared_over_v9_time_ratio', 'shared_over_sequential_time_ratio')}
                      for k, v in report['languages'].items()}, indent=2))


if __name__ == '__main__':
    main()
