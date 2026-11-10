"""Registered TRAIN-only numerical scope/concern transportation diagnostic.

Consumes audited117 saved parents; never replays a head or loads feature values.
All TRAIN rows retained. Fixed-head probabilities and the effective coupled
scope/AR laws are distinct and both retained. Float guards are not certificates.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil

FORMAT = 'roguard-train-transport-bounds-plan-1'
AUDITOR_SHA = '1bb9e5d8b24ce703d857ea14668c97d6ac99c81daf83e53b4316badd635822b9'
TRANSPORT_SHA = '59932304d46d8ee85ca7f376538a2a6d46c138d2e6352cb2694d5343fd86218b'
EXPECTED_BINARY_ERRORS = {'ro': 4, 'uk': 7}
FIELDS = ('subject', 'time', 'assertion', 'source', 'relation', 'concern')
RANGES = ((0, 3), (3, 6), (6, 10), (10, 13), (13, 15), (15, 17))
UNRESOLVED = 'NUMERICAL_BOUNDARY_UNRESOLVED'


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


def same(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def bound_bytes(record):
    require(Path(record['path']).is_absolute(), 'Nonabsolute binding')
    data = Path(record['path']).read_bytes()
    require(hashlib.sha256(data).hexdigest() == record['sha256'], 'Bound bytes changed: ' + record['path'])
    return data


def read_bound(record):
    return json.loads(bound_bytes(record))


def check_files(files):
    for name, record in files.items():
        require(Path(record['path']).is_absolute() and sha(record['path']) == record['sha256'], 'Changed binding: ' + name)


def load_bound_module(name, record):
    spec = importlib.util.spec_from_file_location(name, record['path'])
    module = importlib.util.module_from_spec(spec)
    exec(compile(bound_bytes(record), record['path'], 'exec'), module.__dict__)
    return module


def preflight(plan_path, plan_sha):
    plan = read_bound(dict(path=str(plan_path), sha256=plan_sha))
    require(plan['format'] == FORMAT and plan['registered'] is True
            and plan['authorization'] == 'ROOT_REGISTERED_TRAIN_TRANSPORT_BOUNDS_ONLY'
            and plan['source_sha256'] == sha(__file__)
            and plan['encoder_loaded'] is False and plan['optimizer_run'] is False
            and plan['new_probe_scoring'] is False and plan['qualified'] is False
            and plan['numerical_guard_policy'] == 'TRANSPORT_V2_GUARD_PLUS_CAP_AND_PARENT_CONTRADICTION_UNRESOLVED',
            'Unregistered diagnostic source/scope')
    files = plan['files']
    require(set(files) == {'audit118_plan', 'audit118_source', 'audit118_report', 'audit118_saved_plan',
                          'audit118_saved_source', 'transport_source', 'transport_report', 'transport_review'},
            'Diagnostic input inventory differs')
    check_files(files)
    require(files['audit118_source']['sha256'] == files['audit118_saved_source']['sha256'] == AUDITOR_SHA
            and files['transport_source']['sha256'] == TRANSPORT_SHA
            and files['audit118_plan']['sha256'] == files['audit118_saved_plan']['sha256'], 'Reviewed source/snapshot differs')
    audit_directory = Path(files['audit118_report']['path']).parent
    for key, name in (('audit118_saved_plan', 'audit-plan.json'), ('audit118_saved_source', 'audit-source.py')):
        require(Path(files[key]['path']).resolve() == (audit_directory / name).resolve(), 'Audit snapshot origin differs')
    audit = load_bound_module('frozen_saved_train_auditor118', files['audit118_source'])
    # Reuse only the already reviewed, stdlib preflight. Its numerical run is
    # never invoked; this helper verifies both original populations and ledgers.
    prior_plan, producer, producer_report, languages = audit.preflight(
        Path(files['audit118_plan']['path']), files['audit118_plan']['sha256'])
    verified = read_bound(files['audit118_report'])
    require(verified['format'] == 'roguard-train-oracle-saved-audit-1' and verified['verification'] == 'PASS'
            and verified['audit_plan_sha256'] == files['audit118_plan']['sha256'] and verified['source_sha256'] == AUDITOR_SHA
            and verified['producer_plan_sha256'] == prior_plan['files']['producer_plan']['sha256']
            and verified['producer_source_sha256'] == producer['source_sha256']
            and verified['producer_report_sha256'] == prior_plan['files']['producer_report']['sha256'], 'Audit118 lineage differs')
    for flag in ('encoder_loaded', 'optimizer_run', 'parent_head_replayed', 'new_probe_scoring',
                 'native_serving_parity_claimed', 'exhaustive_reachability_claimed', 'qualified'):
        require(verified[flag] is False, 'Audit118 scope differs: ' + flag)
    require(verified['dev_feature_rows_scored'] == 0 and verified['oracle_not_learned'] is True, 'Audit118 diagnostic scope differs')
    same(verified['input_files'], producer['files'], 'Audit118 input ledger differs')
    same(verified['output_files'], prior_plan['files'], 'Audit118 output ledger differs')
    require(set(verified['languages']) == {'ro', 'uk'}, 'Audit118 missing language')
    for lang in ('ro', 'uk'):
        selected, summary, spec = languages[lang]
        record = verified['languages'][lang]
        require(record['rows_checked'] == len(selected) and record['weights75_sha256'] == audit.WEIGHTS75[lang]
                and record['spec75_sha256'] == audit.SPECS75[lang] and record['fixed_threshold'] == spec['threshold'], 'Audit118 parent/population differs')
        for key, value in summary.items():
            same(record[key], value, 'Audit118 original membership differs: ' + key)
        same(record['groups'], producer_report['languages'][lang]['groups'], 'Audit118 aggregate provenance differs')
        require(record['groups']['all']['parent_binary_errors'] == EXPECTED_BINARY_ERRORS[lang], 'Known TRAIN binary denominator differs')
    transport_report = read_bound(files['transport_report'])
    transport_review = read_bound(files['transport_review'])
    require(transport_report['source_sha256'] == TRANSPORT_SHA
            and transport_report['status'] == 'SYNTHETIC_CHECKS_PASS_NOT_REGISTERED_FOR_REAL_DATA'
            and transport_report['qualified'] is False
            and transport_review['source_sha256'] == TRANSPORT_SHA
            and transport_review['verification'] == 'PASS_SYNTHETIC_BOUNDS_ONLY'
            and transport_review['qualified'] is False, 'Transport source review differs')
    transport = load_bound_module('frozen_transport_bounds_v2', files['transport_source'])
    check_files(files); check_files(producer['files']); check_files(prior_plan['files'])
    require(sha(plan_path) == plan_sha and sha(__file__) == plan['source_sha256'], 'Registration changed during global preflight')
    return plan, audit, prior_plan, producer, verified, languages, transport


def probability_fields(z):
    import numpy as np
    values = {}
    for name, (start, end) in zip(FIELDS, RANGES):
        part = z[start:end].astype(np.float64)
        p = np.exp(part - part.max())
        values[name] = (p / p.sum()).tolist()
    return values


def effective_reachability(bounds, threshold, parent_score, raw):
    """Preserve rawv2 states but never certify cap/served near-boundary conflicts."""
    result = dict(raw)
    result.pop('positive_impossible_by_concern', None)
    guard = bounds['guard']
    p = bounds['positive_mass']
    cap_near = abs(p - threshold) <= guard
    contradictions = []
    if p < threshold and parent_score >= threshold:
        contradictions.append('COMPUTED_CONCERN_CAP_VS_SERVED_POSITIVE')
    if raw['positive'].startswith('IMPOSSIBLE') and parent_score >= threshold:
        contradictions.append('POSITIVE_BOUND_VS_SERVED_POSITIVE')
    if raw['negative'].startswith('IMPOSSIBLE') and parent_score < threshold:
        contradictions.append('NEGATIVE_BOUND_VS_SERVED_NEGATIVE')
    if cap_near and raw['positive'] == 'IMPOSSIBLE_CONCERN_CAP':
        result['positive'] = UNRESOLVED
    if cap_near and raw['negative'] == 'FEASIBLE_CONCERN_CAP':
        # The lower witness may still establish negative feasibility away
        # from the threshold, even when the concern-cap shortcut cannot.
        result['negative'] = ('FEASIBLE_OUTSIDE_GUARD'
                              if bounds['minimum'] + guard < threshold else UNRESOLVED)
    if contradictions:
        # A contradictory implementation comparison supplies no certificate.
        result['positive'] = result['negative'] = UNRESOLVED
    result.update(numerical_boundary=UNRESOLVED in (result['positive'], result['negative']),
                  concern_cap_within_guard=cap_near, served_decision_contradictions=contradictions,
                  concern_cap_impossibility_outside_guard=(p < threshold - guard and not contradictions),
                  non_certified=True)
    return result


def classify(z, row, spec, transport, score, portable, policy):
    before = z.tobytes()
    values = probability_fields(z)
    parent_score = score(z, spec)
    zero = transport.zero_residual_score(values, spec['couplings'], portable, policy)
    require(zero['score'] == parent_score, 'Portable parent score arithmetic differs')
    bounds = transport.greedy_bounds(zero['scope'], values['concern'][1], zero['weights'])
    guard = bounds['guard']
    require(bounds['minimum'] - guard <= parent_score <= bounds['maximum'] + guard, 'Parent score outside numerical bounds guard')
    containment = ('WITHIN_POINT_ESTIMATES' if bounds['minimum'] <= parent_score <= bounds['maximum']
                   else 'ONLY_WITHIN_NONCERTIFIED_GUARD')
    for which in ('min', 'max'):
        table = bounds[which + '_table']
        require(len(table) == 9 and all(len(cell) == 2 for cell in table), 'Incomplete transport witness')
        require(all(v >= 0 for cell in table for v in cell), 'Negative witness cell')
        require(max(abs(math.fsum(cell) - r) for cell, r in zip(table, zero['scope'])) <= guard
                and abs(math.fsum(cell[1] for cell in table) - values['concern'][1]) <= guard
                and abs(math.fsum(cell[0] for cell in table) - values['concern'][0]) <= guard, 'Witness changed frozen marginals')
        key = 'minimum' if which == 'min' else 'maximum'
        require(abs(transport.joint_score(table, zero['weights']) - bounds[key]) <= guard, 'Witness objective differs')
    raw = transport.reachability(bounds, spec['threshold'])
    effective = effective_reachability(bounds, spec['threshold'], parent_score, raw)
    predicted = [int(z[a:b].argmax()) for a, b in RANGES]
    require(z.tobytes() == before, 'Diagnostic changed saved parent logits')
    return dict(parent_fields=predicted, gold_fields=row['fields'], gold_binary=row['gold'],
        parent_score=parent_score, fixed_threshold=spec['threshold'], parent_binary=int(parent_score >= spec['threshold']),
        parent_binary_error=int(parent_score >= spec['threshold']) != row['gold'],
        parent_threshold_distance=parent_score-spec['threshold'],
        parent_score_containment=containment, parent_fields_unchanged=True, raw_logits_bitwise_unchanged=True,
        fixed_head_probabilities=values, coupled_scope=zero['scope'], coupled_assertion_relation=zero['assertion_relation'],
        scope_order=transport.SCOPES, policy_weights=zero['weights'],
        parent_independence_table=zero['table'], independent_parent_joint_score=zero['independent_joint_score'],
        independent_parent_enumerated_score=zero['independent_enumerated_score'],
        maximum_parent_parity_error=zero['maximum_parity_error'], bounds=bounds,
        raw_transport_v2_reachability=raw, effective_reachability=effective,
        gold_compatible_reachability=effective['positive' if row['gold'] else 'negative'],
        computed_concern_below_threshold=values['concern'][1] < spec['threshold'],
        learned_prediction=False, numerical_certificate=False)


def summarize(rows):
    buckets = {}
    for row in rows:
        for name in ('all', 'source_group:' + row['group'], 'gold_binary:' + str(row['gold_binary']),
                     'gold_assertion:' + str(row['gold_fields'][2]), 'parent_binary_error:' + str(int(row['parent_binary_error']))):
            buckets.setdefault(name, []).append(row)
    result = {}
    for name, selected in buckets.items():
        result[name] = dict(rows=len(selected), parent_binary_errors=sum(r['parent_binary_error'] for r in selected),
            gold_compatible_reachability=dict(Counter(r['gold_compatible_reachability'] for r in selected)),
            positive_reachability=dict(Counter(r['effective_reachability']['positive'] for r in selected)),
            negative_reachability=dict(Counter(r['effective_reachability']['negative'] for r in selected)),
            numerical_boundary_rows=sum(r['effective_reachability']['numerical_boundary'] for r in selected),
            computed_concern_below_threshold=sum(r['computed_concern_below_threshold'] for r in selected),
            concern_cap_impossibility_outside_guard=sum(r['effective_reachability']['concern_cap_impossibility_outside_guard'] for r in selected),
            served_decision_contradiction_rows=sum(bool(r['effective_reachability']['served_decision_contradictions']) for r in selected),
            parent_score_containment=dict(Counter(r['parent_score_containment'] for r in selected)),
            maximum_parent_parity_error=max(r['maximum_parent_parity_error'] for r in selected),
            maximum_numerical_guard=max(r['bounds']['guard'] for r in selected),
            maximum_marginal_roundoff=max(r['bounds']['maximum_marginal_roundoff'] for r in selected),
            parent_field_errors=[sum(r['parent_fields'][i] != r['gold_fields'][i] for r in selected) for i in range(6)],
            parent_joint_errors=sum(r['parent_fields'] != r['gold_fields'] for r in selected))
    return result


def run(plan_path, plan_sha, output):
    require(not output.exists(), 'Refusing to overwrite diagnostic output')
    plan, audit, prior_plan, producer, verified, languages, transport = preflight(plan_path, plan_sha)
    own_source = bound_bytes(dict(path=str(Path(__file__).resolve()), sha256=plan['source_sha256']))
    own_plan = bound_bytes(dict(path=str(plan_path), sha256=plan_sha))
    import numpy as np
    from roguard.gemma4_review import _score
    from roguard.native_fields import factorized_review_mass
    from roguard.evidence import _complete_action
    prepared = {}
    for lang in ('ro', 'uk'):
        selected, summary, spec = languages[lang]
        def data(suffix):
            return bound_bytes(prior_plan['files']['output:' + lang + suffix])
        rows_bytes = data('-rows.jsonl')
        retained = [json.loads(line) for line in rows_bytes.splitlines()]
        same(retained, [dict(train_index=i, source_index=index, row=row) for i, (index, row) in enumerate(selected)], 'Saved TRAIN order differs')
        results = [json.loads(line) for line in data('-results.jsonl').splitlines()]
        require(len(results) == len(selected) and audit.digest(results) == verified['languages'][lang]['every_retained_result_canonical_sha256'],
                'Audit118 retained-row digest/denominator differs')
        totals, extra = audit.aggregate(results)
        same(totals, verified['languages'][lang]['groups'], 'Audit118 aggregate recount differs')
        same(extra, verified['languages'][lang]['six_field_groups'], 'Audit118 six-field recount differs')
        parent = np.load(io.BytesIO(data('-parent-logits.npy')), allow_pickle=False)
        require(parent.shape == (len(selected), 19) and parent.dtype == np.float32 and np.isfinite(parent).all(), 'Saved parent array invalid')
        for i, (index, row) in enumerate(selected):
            prior = results[i]
            same({k: prior[k] for k in ('train_index', 'source_index', 'case_id', 'language', 'partition', 'group', 'gold_fields', 'gold_binary')},
                 dict(train_index=i, source_index=index, case_id=row['case_id'], language=lang, partition='train', group=row['group'],
                      gold_fields=row['fields'], gold_binary=row['gold']), 'Prior result metadata differs')
        prepared[lang] = (parent, rows_bytes, results)
    def barrier():
        check_files(plan['files']); check_files(prior_plan['files']); check_files(producer['files']); audit.check_origins(producer)
        require(sha(plan_path) == plan_sha and sha(__file__) == plan['source_sha256'], 'Diagnostic registration changed')
    barrier()  # Both languages, imported scorer and all retained arrays validated before first bound.
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    require(shutil.disk_usage(ancestor).free >= 8 * 1024**3 + 512 * 1024**2, 'Preserve8GiB reserve plus diagnostic output allowance')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source.py').write_bytes(own_source); (output / 'registered-plan.json').write_bytes(own_plan)
    require(sha(output / 'source.py') == plan['source_sha256'] and sha(output / 'registered-plan.json') == plan_sha, 'Retained snapshot differs')
    report = dict(format='roguard-train-transport-bounds-1', complete=False, source_sha256=plan['source_sha256'], plan_sha256=plan_sha,
        audit118_report_sha256=plan['files']['audit118_report']['sha256'], transport_source_sha256=TRANSPORT_SHA,
        files=plan['files'], audit118_files=prior_plan['files'], producer117_files=producer['files'], languages={},
        encoder_loaded=False, optimizer_run=False, parent_head_replayed=False, dev_feature_rows_scored=0,
        new_probe_rows_scored=0, learned_module=False, numerical_certificate=False, native_serving_parity_claimed=False,
        qualified=False, diagnostic_population='ALL original88 TRAIN; no row removed by label, margin, baseline correctness or feasibility',
        limits=['Numerical transportation bounds with non-certified guards; boundary and served/cap contradictions remain unresolved.',
                'Parent logits have registered117 provenance checked by118; this is not a new parent-head/native parity replay.',
                'Scope9 and concern2 effective marginals fixed; raw six-head probabilities/logits unchanged. No learned transport or new accuracy claim.',
                'Gold informs only diagnostic compatibility; extrema depend only on frozen parent probabilities and fixed policy.',
                'A gold-compatible witness is not a learned model or an implementation with certified numerical decisions.'])
    def persist():
        temp = output / 'report.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temp.replace(output / 'report.json')
    persist()
    for lang in ('ro', 'uk'):
        selected, summary, spec = languages[lang]
        parent, rows_bytes, prior_results = prepared[lang]
        (output / (lang + '-rows.jsonl')).write_bytes(rows_bytes)
        require(sha(output / (lang + '-rows.jsonl')) == prior_plan['files']['output:' + lang + '-rows.jsonl']['sha256'],
                'Retained original TRAIN rows differ')
        outcomes = []
        with (output / (lang + '-bounds.jsonl')).open('x') as stream:
            for i, (source_index, row) in enumerate(selected):
                outcome = classify(parent[i], row, spec, transport, _score, factorized_review_mass, _complete_action)
                same(outcome['parent_fields'], prior_results[i]['parent_fields'], 'Saved parent field prediction differs')
                same(outcome['parent_score'], prior_results[i]['parent_score'], 'Saved parent score differs')
                require(outcome['parent_binary'] == prior_results[i]['parent_binary'], 'Saved parent binary differs')
                outcome.update(train_index=i, source_index=source_index, case_id=row['case_id'], language=lang, partition='train', group=row['group'])
                stream.write(canonical(outcome) + '\n'); outcomes.append(outcome)
        persisted = [json.loads(line) for line in (output / (lang + '-bounds.jsonl')).read_text().splitlines()]
        same(persisted, outcomes, 'Retained numerical output differs')
        groups = summarize(persisted)
        require(groups['all']['rows'] == len(selected) and groups['all']['parent_binary_errors'] == EXPECTED_BINARY_ERRORS[lang], 'Full TRAIN error denominator differs')
        same(groups['all']['parent_field_errors'], verified['languages'][lang]['six_field_groups']['all']['parent_field_errors'], 'Parent six-field recount differs')
        same(groups['all']['parent_joint_errors'], verified['languages'][lang]['six_field_groups']['all']['parent_joint_errors'], 'Parent joint recount differs')
        error_rows = [{key: row[key] for key in ('train_index', 'source_index', 'case_id', 'group', 'gold_binary', 'parent_binary',
                      'parent_score', 'gold_compatible_reachability', 'computed_concern_below_threshold', 'effective_reachability', 'bounds')}
                      for row in persisted if row['parent_binary_error']]
        report['languages'][lang] = dict(summary, complete=True, rows_processed=len(persisted), groups=groups,
            existing_parent_binary_error_diagnostics=error_rows, fixed_threshold=spec['threshold'],
            artifacts={name: sha(output / name) for name in (lang + '-rows.jsonl', lang + '-bounds.jsonl')})
        persist()
    barrier()
    require(sha(output / 'source.py') == plan['source_sha256'] and sha(output / 'registered-plan.json') == plan_sha,
            'Retained source/plan changed before completion')
    for record in report['languages'].values():
        for name, expected in record['artifacts'].items():
            require(sha(output / name) == expected, 'Retained artifact changed before completion: ' + name)
    report.update(complete=True, decision='TRAIN_NUMERICAL_TRANSPORT_BOUNDS_ONLY_NOT_LEARNED_PERFORMANCE')
    persist()
    same(json.loads((output / 'report.json').read_text()), report, 'Final report readback differs')
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
