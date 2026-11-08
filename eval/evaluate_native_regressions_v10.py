"""Fresh published-v9 and selected-candidate serving inference after DEV passes.

Exposed synthetic regressions only. No fitting, threshold selection, sealed TEST,
reviewer admission or release qualification is performed by this program.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time

from eval.native_regression_inputs_v10 import (POPULATIONS, PUBLISHED_RUNTIME, SELECTOR_SHA,
    load_inputs, require, revalidate_admission, verify_runtime)
from roguard.trajectory_data import file_sha, write_jsonl
from roguard.trajectory_evaluation import checkpoint_id, evaluate, paired_compare


def world_from_logits(values):
    from roguard.evidence import ONTOLOGY
    from roguard.native_fields import FIELDS, SIZES
    import math
    require(len(values) == 19 and all(math.isfinite(x) for x in values), 'Invalid retained field logits')
    offset = 0; world = {}
    for field, size in zip(FIELDS, SIZES):
        section = values[offset:offset+size]
        world[field] = ONTOLOGY[field][max(range(size), key=lambda i: section[i])]
        offset += size
    return world


def predict_records(inputs, predict, identity, *, progress=None):
    """Memoize identical serving inputs while retaining all scoring denominators."""
    memo, records = {}, []
    for item in inputs:
        key = item['inference_input_digest']
        if item['unsupported']:
            raw, logits = dict(predicted=None, score=None), None
        else:
            if key not in memo:
                memo[key] = predict(item['inference'])
            raw, logits = memo[key]
        value, score = raw['predicted'], raw['score']
        require(value is None or type(value) is int and value in (0, 1), 'Invalid serving decision')
        require((value is None) == (score is None) == (logits is None), 'Incomplete serving prediction')
        require(score is None or type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                'Invalid serving score')
        record = {k: item[k] for k in ('population', 'checkpoint_id', 'language', 'focus',
                                     'input_digest', 'inference_input_digest', 'evidence')}
        record.update(predicted=value, score=score, field_logits=logits, model_identity=identity)
        if logits is not None:
            record['interpretation_set'] = dict(hypotheses=[world_from_logits(logits)],
                reason='UNCALIBRATED_ARGMAX_DIAGNOSTIC', coverage_assumptions_verified=False,
                outside_ontology_ruled_out=False)
        records.append(record)
        if progress and (len(records) % 100 == 0 or len(records) == len(inputs)):
            progress(dict(rows=len(records), total_rows=len(inputs), unique_serving_calls=len(memo)))
    return records, len(memo)


def infer_model(component, inputs, progress):
    """Use the unchanged public engine; tap only its head output for diagnostics."""
    import numpy as np
    import mlx.nn as nn
    from roguard.gemma4_review import _Engine, _score
    component._engine = engine = _Engine(*component._args)
    mx = engine.mx; lang = component.language
    class Tap(nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner; self.logits = None
        def __call__(self, features):
            values = self.inner(features); mx.eval(values)
            self.logits = np.asarray(values)[0].copy().tolist()
            return values
    tap = Tap(engine.model.heads[lang]); engine.model.heads[lang] = tap
    def predict(payload):
        tap.logits = None
        result = component.predict(payload)
        logits = tap.logits
        if logits is not None:
            require(abs(_score(logits, component._spec)-result['score']) <= 1e-10,
                    'Captured logits disagree with public serving score')
        return result, logits
    try:
        return predict_records(inputs, predict, component.identity, progress=progress)
    finally:
        engine.model.heads[lang] = tap.inner
        component._engine = None
        del engine, tap
        gc.collect(); mx.clear_cache()


def assess(cases, baseline, candidate):
    """Missing field annotations remain missing, not manufactured successes."""
    baseline_report = evaluate(cases, baseline, bootstrap=0)
    candidate_report = evaluate(cases, candidate, bootstrap=0)
    paired = paired_compare(cases, baseline, candidate)
    old = {r['checkpoint_id']: r for r in baseline}; new = {r['checkpoint_id']: r for r in candidate}
    totals = dict(rows=0, candidate_errors=0, v9_errors=0, fixed_v9=0, introduced_v9=0,
                  annotated_worlds=0, joint_argmax_errors=0, unannotated_worlds=0)
    groups = {}
    for case in cases:
        for point in case['checkpoints']:
            key = checkpoint_id(case, point); a, b = old[key], new[key]
            require(a['inference_input_digest'] == b['inference_input_digest'], 'Compared memory views differ')
            world = point.get('world')
            values = dict(rows=1, candidate_errors=int(b['predicted'] != point['review']),
                v9_errors=int(a['predicted'] != point['review']),
                fixed_v9=int(b['predicted'] == point['review'] and a['predicted'] != point['review']),
                introduced_v9=int(b['predicted'] != point['review'] and a['predicted'] == point['review']),
                annotated_worlds=int(world is not None), unannotated_worlds=int(world is None),
                joint_argmax_errors=int(world is not None and
                    (b['field_logits'] is None or world_from_logits(b['field_logits']) != world)))
            group = groups.setdefault(case['language'], {k: 0 for k in totals})
            for k, v in values.items(): totals[k] += v; group[k] += v
    return dict(totals=totals, groups=groups, baseline=baseline_report, candidate=candidate_report,
                paired=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private_root', 'base', 'selector', 'study'):
        parser.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    parser.add_argument('--arm', choices=('control', 'treatment'), required=True)
    parser.add_argument('--producer-pid', type=int, required=True)
    parser.add_argument('--preflight', action='store_true')
    a = parser.parse_args()
    require(not a.study.exists(), 'Refusing to overwrite native regression study')
    proof, proof_sha = revalidate_admission(a.root, a.private_root, a.arm, a.producer_pid)
    verify_runtime(a.root)
    cases, inputs = load_inputs(a.root, a.selector)
    from training.register_matched_role_attention_v10 import OUTPUTS
    selected_study = a.root/OUTPUTS[a.arm]['study']; selected = a.private_root/OUTPUTS[a.arm]['private']
    fixed = {a.root/name: sha for name, sha in PUBLISHED_RUNTIME.items()}
    fixed.update({a.root/path: sha for path, sha, _ in POPULATIONS.values()})
    fixed[a.selector] = SELECTOR_SHA
    for relative in ('eval/native_regression_inputs_v10.py', 'eval/evaluate_native_regressions_v10.py',
                     'eval/run_trajectory_v10.py', 'src/roguard/trajectory_model.py',
                     'src/roguard/trajectory_memory.py', 'src/roguard/trajectory_neural.py',
                     'src/roguard/trajectory_data.py', 'src/roguard/trajectory_evaluation.py'):
        fixed[a.root/relative] = file_sha(a.root/relative)
    for lang in ('ro', 'uk'):
        path = selected/(lang+'-selected.safetensors')
        require(file_sha(path) == proof['arms'][a.arm]['languages'][lang]['artifacts'][path.name],
                'Selected adapter changed after admission')
        fixed[path] = file_sha(path)
        path = selected_study/(lang+'-review.json'); fixed[path] = file_sha(path)
    if a.preflight:
        print(json.dumps(dict(preflight='PASS', arm=a.arm, checkpoints=len(inputs),
            unsupported=sum(r['unsupported'] for r in inputs), encoder_loaded=False, model_qualified=False)))
        return
    a.study.mkdir(parents=True)
    def write(name, value):
        (a.study/name).write_text(json.dumps(value, indent=2)+'\n')
    from eval.run_trajectory_v10 import LocalFrozenV9
    from roguard.gemma4_review import Gemma4ReviewModel
    frozen = LocalFrozenV9(a.base, a.root/'research/v9-stabilization/reported-tail-fit-196', a.study/'v9-binding')
    protocol = dict(format='roguard-native-regression-1', arm=a.arm, admission_sha256=proof_sha,
        inputs={name: sha for name, (_, sha, _) in POPULATIONS.items()}, selector_sha256=SELECTOR_SHA,
        runtime_sources=PUBLISHED_RUNTIME, checkpoints=len(inputs), memory_budget=1200,
        preserve_complete_turns=True, fresh_v9_and_candidate=True, persistent_body_cache=False,
        sources={str(p.relative_to(a.root)): sha for p, sha in fixed.items() if p.is_relative_to(a.root)
                 and p.suffix == '.py'},
        candidate_adapters={lang: fixed[selected/(lang+'-selected.safetensors')] for lang in ('ro', 'uk')},
        candidate_specs={lang: fixed[selected_study/(lang+'-review.json')] for lang in ('ro', 'uk')},
        memoization='Identical selected inputs share one serving result; all checkpoints remain scored.',
        fitting_performed=False, threshold_selection_performed=False, held_out_test_opened=False,
        independent_validation=False, stable_release_ready=False, end_to_end_latency_measured=False)
    write('protocol.json', protocol)
    outputs = {'baseline': [], 'candidate': []}; forwards = {}; started = time.monotonic()
    for lang in ('ro', 'uk'):
        subset = [item for item in inputs if item['language'] == lang]
        for kind in ('baseline', 'candidate'):
            component = (frozen.models[lang] if kind == 'baseline' else Gemma4ReviewModel(
                selected_study/(lang+'-review.json'), selected/(lang+'-selected.safetensors'), a.base,
                expected_sha256=fixed[selected_study/(lang+'-review.json')]))
            def progress(value):
                value.update(language=lang, component=kind, seconds=time.monotonic()-started)
                write('progress.json', value); print(json.dumps(value), flush=True)
            records, calls = infer_model(component, subset, progress)
            outputs[kind].extend(records); forwards[lang+':'+kind] = calls
            write_jsonl(a.study/(kind+'.jsonl'), outputs[kind])
            del component; gc.collect()
    populations = {name: assess(rows, *[[r for r in outputs[k] if r['population'] == name]
                                      for k in ('baseline', 'candidate')]) for name, rows in cases.items()}
    require(all(file_sha(path) == sha for path, sha in fixed.items()), 'Bound regression assets changed during inference')
    failed = any(p['totals'][k] for p in populations.values()
                 for k in ('candidate_errors', 'introduced_v9', 'joint_argmax_errors'))
    report = dict(complete=True, populations=populations, fresh_serving_calls=forwards,
        decision='REJECT' if failed else 'EXPOSED_REGRESSION_PASS_ONLY',
        prediction_sha256={k: file_sha(a.study/(k+'.jsonl')) for k in outputs},
        seconds=time.monotonic()-started, held_out_test_opened=False, independent_validation=False,
        stable_release_ready=False, reviewers_admitted=False,
        limitations='Synthetic exposed engineering populations. Singleton argmax is uncalibrated; '
                    'fresh44 has no field gold. Response, independent calibration, sealed evaluation, '
                    'reviewer benefit and installed-runtime qualification remain separate requirements.')
    write('report.json', report)
    print(json.dumps(dict(complete=True, decision=report['decision'], stable_release_ready=False)))


if __name__ == '__main__':
    main()
