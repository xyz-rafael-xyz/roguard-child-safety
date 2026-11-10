"""Independent saved-output audit of registered TRAIN oracle feasibility.

Never loads features/weights numerically, imports producer helpers, replays a
head, or runs an encoder/optimizer. Replays only the registered deterministic
adapter on saved parent logits. A PASS is not native parity, learned quality,
or exhaustive conservative reachability.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
from importlib.metadata import version
import io
import json
import math
from pathlib import Path
import sys

FORMAT = 'roguard-train-oracle-saved-audit-plan-1'
PRODUCER_SHA = 'b1c57ecde2b80feb20be71e077e53604e67e8b7e22786f4eb0232ad88b31a847'
ADAPTER_SHA = '12754dca395eff37f087a41b049cdea0f966427dc558d72d476f6de37a992fa2'
PROJECTION_SHA = 'aec96156c0c029c56c490c89802bc22b57e7c3cbe8cbd06b19e11276e99a2959'
GELU_SHA = 'b89b4f643ff7dc2dcfdaab70fe11b0a7641de17740a1fc0ae8713006bf09850e'
COUNTS = {'ro': (17376, 13174), 'uk': (20390, 15755)}
WEIGHTS75 = {'ro': '1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3',
             'uk': '466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}
SPECS75 = {'ro': '5fdeae97843d873755ed41a02453b5c29823fd8028508c3e99d1586175c8f4a9',
           'uk': 'bbf3fea8ee298c40d86701851fe045352ea1a924ba5287dd52b70e59ce25de27'}
THRESHOLDS = {'ro': .7517333793124337, 'uk': .9274248149759536}
SIZES = (3, 3, 4, 3, 2, 2)
RANGES = ((0, 3), (3, 6), (6, 10), (10, 13), (13, 15), (15, 17))
OTHER = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def same(actual, expected, message):
    # JSON canonical comparison distinguishes bools from ints and includes all
    # keys. No tolerance hides per-row float or adapter-observation changes.
    require(canonical(actual) == canonical(expected), message)


def bound_bytes(record):
    require(Path(record['path']).is_absolute(), 'Nonabsolute binding')
    data = Path(record['path']).read_bytes()
    require(hashlib.sha256(data).hexdigest() == record['sha256'], 'Bound bytes changed: ' + record['path'])
    return data


def read_bound(record):
    return json.loads(bound_bytes(record))


def check_files(files):
    for key, record in files.items():
        require(Path(record['path']).is_absolute() and sha(record['path']) == record['sha256'], 'Binding changed: ' + key)


def original_train(record, language, expected):
    rows = [json.loads(line) for line in bound_bytes(record).splitlines()]
    chosen = []
    seen = set()
    for index, row in enumerate(rows):
        require(row.get('partition') in ('train', 'dev'), 'Unexpected original88 partition')
        if row['partition'] == 'dev':
            continue  # Deliberately never inspect DEV labels.
        require(row.get('language') == language and isinstance(row.get('case_id'), str)
                and row['case_id'] and row['case_id'] not in seen, 'Invalid TRAIN identity')
        seen.add(row['case_id'])
        require(isinstance(row.get('fields'), list) and len(row['fields']) == 6
                and all(type(v) is int and 0 <= v < n for v, n in zip(row['fields'], SIZES))
                and type(row.get('gold')) is int and row['gold'] in (0, 1), 'Invalid TRAIN labels')
        require(isinstance(row.get('group'), str) and row['group'], 'Invalid TRAIN group')
        chosen.append([index, row])
    require((len(rows), len(chosen)) == COUNTS[language], 'Original88 TRAIN population changed')
    summary = dict(total_metadata_rows=len(rows), train_rows=len(chosen),
                   train_order_sha256=digest(chosen), train_groups=dict(Counter(r['group'] for _, r in chosen)))
    same(summary, expected, 'Producer TRAIN partition/order/group binding changed')
    summary['dev_metadata_rows_skipped'] = len(rows) - len(chosen)
    return chosen, summary


def check_origins(producer):
    root = Path(producer['project_root']).resolve()
    bound = {Path(v['path']).resolve(): v['sha256'] for v in producer['files'].values()}
    for name, module in list(sys.modules.items()):
        if name == 'roguard' or name.startswith('roguard.'):
            expected = root / 'src' / Path(*name.split('.'))
            expected = expected / '__init__.py' if expected.is_dir() else expected.with_suffix('.py')
            actual = Path(module.__file__).resolve()
            require(actual == expected.resolve() and actual in bound and sha(actual) == bound[actual],
                    'Unbound portable scorer origin: ' + name)


def preflight(plan_path, plan_sha):
    plan_record = {'path': str(plan_path), 'sha256': plan_sha}
    plan = read_bound(plan_record)
    require(plan['format'] == FORMAT and plan['registered'] is True
            and plan['authorization'] == 'ROOT_REGISTERED_SAVED_TRAIN_ORACLE_AUDIT_ONLY'
            and plan['source_sha256'] == sha(__file__)
            and plan['encoder_loaded'] is False and plan['optimizer_run'] is False
            and plan['new_probe_scoring'] is False and plan['qualified'] is False,
            'Unregistered audit source/scope')
    files = plan['files']
    basenames = {'source.py', 'registered-plan.json'} | {
        lang + suffix for lang in ('ro', 'uk') for suffix in
        ('-rows.jsonl', '-results.jsonl', '-parent-logits.npy', '-oracle-logits.npy')}
    require(set(files) == {'producer_plan', 'producer_source', 'producer_report'} |
            {'output:' + name for name in basenames}, 'Audit ledger inventory differs')
    output = Path(plan['producer_output'])
    require(output.is_absolute(), 'Nonabsolute producer output')
    for name in basenames:
        require(Path(files['output:' + name]['path']).resolve() == (output / name).resolve(), 'Output path binding differs')
    require(Path(files['producer_report']['path']).resolve() == (output / 'report.json').resolve(), 'Producer report path differs')
    check_files(files)
    producer = read_bound(files['producer_plan'])
    require(producer['format'] == 'roguard-train-oracle-feasibility-plan-1'
            and producer['registered'] is True
            and producer['authorization'] == 'ROOT_REGISTERED_TRAIN_ORACLE_FEASIBILITY_ONLY'
            and producer['source_sha256'] == PRODUCER_SHA == files['producer_source']['sha256']
            and producer['risk_tolerance'] == 1e-12 and producer['batch_size'] == 256
            and producer['gelu_function_sha256'] == GELU_SHA
            and producer['optimizer_run'] is False and producer['new_probe_scoring'] is False,
            'Producer source/plan contract differs')
    require(files['output:source.py']['sha256'] == PRODUCER_SHA
            and files['output:registered-plan.json']['sha256'] == files['producer_plan']['sha256'], 'Producer snapshots differ')
    require(set(producer['languages']) == {'ro', 'uk'}, 'Missing language')
    inputs = producer['files']
    check_files(inputs)
    required = {'serving_adapter', 'probability_projection', 'feature_audit89', 'warm_start90_report',
                'adapter_report', 'independent_adapter_review'} | {
                    lang + suffix for lang in ('ro', 'uk') for suffix in ('_rows', '_features', '_spec75', '_weights75')}
    require(required <= set(inputs), 'Incomplete producer lineage')
    root = Path(producer['project_root'])
    ledger_paths = {Path(r['path']).resolve() for r in inputs.values()}
    require((root / 'src/roguard').is_dir() and all(p.resolve() in ledger_paths for p in (root / 'src/roguard').rglob('*.py')),
            'Incomplete serving source closure')
    require(inputs['serving_adapter']['sha256'] == ADAPTER_SHA and inputs['probability_projection']['sha256'] == PROJECTION_SHA,
            'Unreviewed deterministic adapter/projection')
    adapter_path = Path(inputs['serving_adapter']['path'])
    require(Path(inputs['probability_projection']['path']).resolve() ==
            (adapter_path.parent.parent / 'coupling-projection-stage0-v2/projection.py').resolve(), 'Projection import path differs')
    require(version('numpy') == producer['numerical_versions']['numpy'] == plan['numpy_version'], 'NumPy version differs')
    audit89 = read_bound(inputs['feature_audit89'])
    warm90 = read_bound(inputs['warm_start90_report'])
    adapter_report = read_bound(inputs['adapter_report'])
    adapter_review = read_bound(inputs['independent_adapter_review'])
    require(audit89['verification'] == 'PASS' and audit89['qualified'] is False
            and warm90['complete'] is True and warm90['qualified'] is False, 'Original88/75 lineage incomplete')
    require(adapter_report['verification'] == 'PASS' and adapter_report['adapter_sha256'] == ADAPTER_SHA
            and adapter_report['original_score_tolerance'] == 1e-12 and adapter_report['tolerance_relaxed'] is False
            and adapter_review['verification'] == 'PASS_SCOPED_NUMERICAL_GUARD' and adapter_review['qualified'] is False,
            'Adapter test/review contract differs')
    report = read_bound(files['producer_report'])
    require(report['format'] == 'roguard-train-oracle-feasibility-1' and report['complete'] is True
            and report['plan_sha256'] == files['producer_plan']['sha256'] and report['source_sha256'] == PRODUCER_SHA
            and report['decision'] == 'TRAIN_ORACLE_REACHABILITY_ONLY_NOT_LEARNED_PERFORMANCE'
            and report['oracle_not_learned'] is True and report['qualified'] is False
            and report['encoder_loaded'] is False and report['optimizer_run'] is False
            and report['native_serving_parity_claimed'] is False
            and type(report['dev_feature_rows_scored']) is int and report['dev_feature_rows_scored'] == 0
            and type(report['new_probe_rows_scored']) is int and report['new_probe_rows_scored'] == 0
            and report['parent_threshold_diagnostic_band'] == 1e-3, 'Producer report scope/completion differs')
    require(set(report['languages']) == {'ro', 'uk'}, 'Producer report language set differs')
    languages = {}
    for language in ('ro', 'uk'):
        require(inputs[language + '_weights75']['sha256'] == WEIGHTS75[language]
                and inputs[language + '_spec75']['sha256'] == SPECS75[language]
                and warm90['languages'][language]['artifacts'][language + '-selected.safetensors'] == WEIGHTS75[language],
                'Not frozen75 weights/specification')
        for suffix, extension in (('rows', '.jsonl'), ('features', '.npy')):
            require(audit89['artifacts'][language + '-' + suffix + extension] == inputs[language + '_' + suffix]['sha256'],
                    'Original88 audited input differs')
        spec = read_bound(inputs[language + '_spec75'])
        require(spec['language'] == language and spec['mode'] == 'coupled_policy'
                and spec['threshold'] == THRESHOLDS[language] and spec['adapter_sha256'] == WEIGHTS75[language],
                'Frozen75 policy/threshold differs')
        selected, summary = original_train(inputs[language + '_rows'], language, producer['languages'][language])
        record = report['languages'][language]
        require(record['complete'] is True and record['rows_processed'] == len(selected), 'Incomplete language output')
        for key, value in summary.items():
            same(record[key], value, 'Producer population summary differs: ' + key)
        expected_artifacts = {name: files['output:' + name]['sha256'] for name in basenames if name.startswith(language + '-')}
        same(record['artifacts'], expected_artifacts, 'Producer artifact ledger differs')
        languages[language] = (selected, summary, spec)
    return plan, producer, report, languages


def row_result(z, oracle, row, source_index, train_index, language, spec, adapter, score):
    import numpy as np
    before_fields = [int(z[a:b].argmax()) for a, b in RANGES]
    after_fields = [int(oracle[a:b].argmax()) for a, b in RANGES]
    before, after, gold = before_fields[2], after_fields[2], row['fields'][2]
    pair = before in (0, 2) and gold in (0, 2)
    eligible = pair and before != gold
    if eligible:
        # Reproduce only this specified oracle target, from an independently
        # computed float64 assertion softmax, not producer observations.
        p = z[6:10].astype(np.float64)
        p = np.exp(p - p.max()); p = (p / p.sum()).tolist()
        p[0], p[2] = p[2], p[0]
        adapted = adapter.adapt(z.copy(), spec, target_assertion=p, score_tolerance=1e-12)
        expected = adapted['logits']
        status, reason = adapted['status'], adapted['fallback_reason']
        observation = {key: value for key, value in adapted.items() if key != 'logits'}
    else:
        expected = z
        status = 'PARENT_ALREADY_CORRECT' if before == gold else 'OUT_OF_SCOPE_ASSERTION_PAIR'
        reason = observation = None
    require(expected.dtype == oracle.dtype and expected.tobytes() == oracle.tobytes(), 'Specified deterministic adapter output differs')
    identity = z.tobytes() == oracle.tobytes()
    require(z[OTHER].tobytes() == oracle[OTHER].tobytes(), 'Other15 logits changed')
    if not eligible or status == 'EXACT_PARENT_FALLBACK':
        require(identity, 'Untouched/fallback row lost bitwise parent identity')
    parent_score, oracle_score = score(z, spec), score(oracle, spec)
    require(math.isfinite(parent_score) and math.isfinite(oracle_score)
            and abs(oracle_score - parent_score) <= 1e-12
            and (parent_score >= spec['threshold']) == (oracle_score >= spec['threshold']), 'Score/decision invariant differs')
    return dict(train_index=train_index, source_index=source_index, case_id=row['case_id'], language=language,
                partition='train', group=row['group'], gold_fields=row['fields'], gold_binary=row['gold'],
                parent_fields=before_fields, oracle_fields=after_fields, parent_assertion=before, oracle_assertion=after,
                eligible_au_disagreement=eligible, au_pair_in_scope=pair, status=status, fallback_reason=reason,
                parent_score=parent_score, oracle_score=oracle_score, score_difference=oracle_score-parent_score,
                parent_signed_distance_to_threshold=parent_score-spec['threshold'],
                parent_absolute_distance_to_threshold=abs(parent_score-spec['threshold']),
                parent_within_1e_3_threshold_band=abs(parent_score-spec['threshold']) <= 1e-3,
                parent_binary=int(parent_score >= spec['threshold']), oracle_binary=int(oracle_score >= spec['threshold']),
                corrected_assertion=before != gold and after == gold, introduced_assertion_error=before == gold and after != gold,
                other15_logits_bitwise_unchanged=True, full_parent_identity=identity,
                adapter_observation=observation, oracle_not_learned=True)


def aggregate(rows):
    groups = {}
    for r in rows:
        gold = r['gold_fields'][2]
        for group in ('all', 'gold_assertion:' + str(gold), 'source_group:' + r['group']):
            groups.setdefault(group, []).append(r)
    result, extended = {}, {}
    for group, values in groups.items():
        result[group] = dict(rows=len(values), eligible=sum(r['eligible_au_disagreement'] for r in values),
            corrected=sum(r['corrected_assertion'] for r in values), introduced=sum(r['introduced_assertion_error'] for r in values),
            parent_assertion_errors=sum(r['parent_assertion'] != r['gold_fields'][2] for r in values),
            oracle_assertion_errors=sum(r['oracle_assertion'] != r['gold_fields'][2] for r in values),
            parent_binary_errors=sum(r['parent_binary'] != r['gold_binary'] for r in values),
            oracle_binary_errors=sum(r['oracle_binary'] != r['gold_binary'] for r in values),
            maximum_score_difference=max(abs(r['score_difference']) for r in values),
            minimum_parent_threshold_distance=min(r['parent_absolute_distance_to_threshold'] for r in values),
            parent_within_1e_3_threshold_band=sum(r['parent_within_1e_3_threshold_band'] for r in values),
            statuses=dict(Counter(r['status'] for r in values)),
            fallback_reasons=dict(Counter(r['fallback_reason'] for r in values if r['fallback_reason'])))
        extra = dict(rows=len(values), full_parent_identity=sum(r['full_parent_identity'] for r in values),
                     au_pair_in_scope=sum(r['au_pair_in_scope'] for r in values))
        for mode in ('parent', 'oracle'):
            extra[mode + '_field_errors'] = [sum(r[mode + '_fields'][i] != r['gold_fields'][i] for r in values) for i in range(6)]
            extra[mode + '_joint_errors'] = sum(r[mode + '_fields'] != r['gold_fields'] for r in values)
            extra[mode + '_joint_or_binary_errors'] = sum(r[mode + '_fields'] != r['gold_fields'] or r[mode + '_binary'] != r['gold_binary'] for r in values)
        extended[group] = extra
    return result, extended


def run(plan_path, plan_sha, output):
    require(not output.exists(), 'Refusing to overwrite audit output')
    plan, producer, producer_report, languages = preflight(plan_path, plan_sha)
    audit_plan_bytes = bound_bytes({'path': str(plan_path), 'sha256': plan_sha})
    audit_source_bytes = bound_bytes({'path': str(Path(__file__).resolve()), 'sha256': plan['source_sha256']})
    import numpy as np
    from roguard.gemma4_review import _score
    check_origins(producer)
    record = producer['files']['serving_adapter']
    module_spec = importlib.util.spec_from_file_location('saved_oracle_audit_adapter', record['path'])
    adapter = importlib.util.module_from_spec(module_spec)
    exec(compile(bound_bytes(record), record['path'], 'exec'), adapter.__dict__)
    check_origins(producer)
    check_files(producer['files']); check_files(plan['files'])
    require(sha(plan_path) == plan_sha and sha(__file__) == plan['source_sha256'],
            'Audit registration changed before saved-row replay')
    report = dict(format='roguard-train-oracle-saved-audit-1', verification='PASS',
        audit_plan_sha256=plan_sha, source_sha256=sha(__file__),
        producer_plan_sha256=plan['files']['producer_plan']['sha256'], producer_source_sha256=PRODUCER_SHA,
        producer_report_sha256=plan['files']['producer_report']['sha256'],
        input_files=producer['files'], output_files=plan['files'], languages={},
        encoder_loaded=False, optimizer_run=False, parent_head_replayed=False, new_probe_scoring=False,
        dev_feature_rows_scored=0, native_serving_parity_claimed=False, exhaustive_reachability_claimed=False,
        oracle_not_learned=True, qualified=False,
        limitations=['Saved parent logits are trusted only as registered producer outputs; no independent head/native replay.',
                     'One fixed gold-informed A/U swap target only; no learned-performance or full-reachability claim.',
                     'All TRAIN rows retained, including unsupported pairs, preexisting errors, near-threshold rows and fallbacks.'])
    for language in ('ro', 'uk'):
        selected, summary, spec = languages[language]
        def payload(suffix):
            return bound_bytes(plan['files']['output:' + language + suffix])
        retained = [json.loads(line) for line in payload('-rows.jsonl').splitlines()]
        expected_rows = [dict(train_index=i, source_index=index, row=row) for i, (index, row) in enumerate(selected)]
        same(retained, expected_rows, 'Saved TRAIN membership/order differs')
        results = [json.loads(line) for line in payload('-results.jsonl').splitlines()]
        require(len(results) == len(selected), 'Missing/extra retained result rows')
        parent = np.load(io.BytesIO(payload('-parent-logits.npy')), allow_pickle=False)
        oracle = np.load(io.BytesIO(payload('-oracle-logits.npy')), allow_pickle=False)
        for array in (parent, oracle):
            require(array.dtype == np.float32 and array.shape == (len(selected), 19)
                    and np.isfinite(array).all(), 'Saved logits shape/dtype/finiteness differs')
        reconstructed = []
        for i, (index, row) in enumerate(selected):
            expected = row_result(parent[i], oracle[i], row, index, i, language, spec, adapter, _score)
            same(results[i], expected, 'Saved row result differs: ' + language + ':' + str(i))
            reconstructed.append(expected)
        totals, six_fields = aggregate(reconstructed)
        same(totals, producer_report['languages'][language]['groups'], 'Producer aggregate counts differ: ' + language)
        report['languages'][language] = dict(summary, rows_checked=len(selected),
            deterministic_adapter_rows_replayed=totals['all']['eligible'],
            every_retained_result_canonical_sha256=digest(reconstructed), groups=totals, six_field_groups=six_fields,
            weights75_sha256=WEIGHTS75[language], spec75_sha256=SPECS75[language], fixed_threshold=spec['threshold'])
    # Recheck inputs, outputs, imported origins and auditor registration before
    # publishing evidence. Files were decoded from hash-verified bytes.
    check_files(producer['files']); check_files(plan['files']); check_origins(producer)
    require(sha(plan_path) == plan_sha and sha(__file__) == plan['source_sha256'], 'Audit registration changed during run')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audit-source.py').write_bytes(audit_source_bytes)
    (output / 'audit-plan.json').write_bytes(audit_plan_bytes)
    require(sha(output / 'audit-source.py') == plan['source_sha256']
            and sha(output / 'audit-plan.json') == plan_sha, 'Retained audit snapshot differs')
    with (output / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    require(json.loads((output / 'report.json').read_bytes()) == report, 'Audit evidence readback differs')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.plan_sha256, args.output)


if __name__ == '__main__':
    main()
