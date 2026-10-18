"""Evaluate qualified semantic-head candidates from an independently installed wheel.

Run by absolute path outside the repository, with PYTHONPATH pointing only to
the wheel installation. A plan pins two installed-runtime reports and all model
assets. This program refuses to open reserved messages before both language
candidates pass their existing development, calibration and runtime checks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path

if __package__:
    from .reserved_metrics_v9 import counts, input_key, unique_records
else:
    from reserved_metrics_v9 import counts, input_key, unique_records

PROBES = {
    'contrast': ('contrast-probe.jsonl', '7bf47bd773d5aceed684ad0aaeced6b85dc7d1602acfe17312a47073fe77e42f'),
    'families_original': ('context-families-01/test.jsonl', '3786bfdc01a0421668df04eae8263488d4cb744bffb54192d647daf48d4fa40d'),
    'families_lexical_v2': ('context-families-01/test-lexical-v2.jsonl', '290cbb6dd7ccdccdf9c4371e4569b49831fbcb8eff5822d1dba5104819669b38'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_report(entry):
    path = Path(entry['runtime_report'])
    if digest(path) != entry['runtime_report_sha256']:
        raise ValueError('Installed runtime evidence changed')
    report = json.loads(path.read_text())
    from roguard.discrete_fields import CONTRACT_DIGEST
    if (report.get('format') != 'roguard-installed-discrete-fields-replay-v1'
            or report.get('contract_digest') != CONTRACT_DIGEST
            or report.get('complete') is not True or report.get('parity_pass') is not True
            or report.get('accuracy_gates_pass') is not True
            or report.get('artifact_sha256') != entry['artifact_sha256']
            or report.get('native_artifact_sha256') != entry['native_artifact_sha256']
            or report.get('selection_sha256') != entry['selection_sha256']
            or set(report.get('splits', {})) != {'dev', 'cal'}):
        raise ValueError('Candidate lacks passing installed development and calibration evidence')
    for split in report['splits'].values():
        if set(split['counts']) != {'formal', 'conversational'}:
            raise ValueError('Required accuracy strata are missing')
        if not split['parity_pass'] or not split['accuracy_gates_pass'] or split['abstentions']:
            raise ValueError('A required installed split failed')
        if (split['decision_changes'] or split['interpretation_changes']
                or not 0 <= split['maximum_field_error'] <= 1e-4):
            raise ValueError('Installed runtime parity failed')
        for value in split['counts'].values():
            if any(type(value.get(k)) is not int or value[k] < 0 for k in
                   ('positive', 'negative', 'missed', 'false_reviews', 'abstentions')):
                raise ValueError('Invalid accuracy counts')
            if (not value['positive'] or not value['negative'] or value['abstentions']
                    or value['missed'] > value['positive'] or value['false_reviews'] > value['negative']
                    or value['missed']*100 > value['positive']
                    or value['false_reviews']*20 > value['negative'] or value.get('passes') is not True):
                raise ValueError('Accuracy counts failed unchanged gates')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'installed-root', 'probe-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.plan_sha256:
        raise ValueError('Candidate selection plan changed')
    plan = json.loads(args.plan.read_text())
    for name in ('verify_reserved_discrete_fields_v9.py', 'reserved_metrics_v9.py'):
        if digest(Path(__file__).with_name(name)) != plan['verifier_source_sha256'][name]:
            raise ValueError('Reserved verifier differs from frozen plan')
    if set(plan['candidates']) != {'ro', 'uk'}:
        raise ValueError('Both language candidates required before reserved evaluation')
    reports = {language: checked_report(entry) for language, entry in plan['candidates'].items()}
    import roguard.native_fields as native
    import roguard.discrete_fields as discrete
    if any(report.get('language') != language for language, report in reports.items()):
        raise ValueError('Installed report language differs from candidate')
    from roguard.evidence import disposition
    import torch
    torch.set_num_threads(4)
    if args.installed_root.resolve() not in Path(native.__file__).resolve().parents:
        raise ValueError('Runtime was not imported from the isolated wheel installation')
    if any(report['runtime_source_sha256'] != digest(discrete.__file__) for report in reports.values()):
        raise ValueError('Runtime source differs from installed replay')
    for name, expected in plan['runtime_source_sha256'].items():
        module = importlib.import_module(name)
        if (args.installed_root.resolve() not in Path(module.__file__).resolve().parents
                or digest(module.__file__) != expected):
            raise ValueError('Installed dependency differs from frozen runtime')
    for entry in plan['candidates'].values():
        for path, key in (('artifact', 'artifact_sha256'), ('native_artifact', 'native_artifact_sha256'),
                          ('selection', 'selection_sha256')):
            if digest(entry[path]) != entry[key]:
                raise ValueError('Selected semantic-head artifact changed')
    # Persist the candidate choice and evidence before opening held-out inputs.
    args.output.mkdir(parents=True, exist_ok=False)
    selection = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=digest(args.plan),
                     candidates={language: {key: entry[key] for key in ('artifact_sha256', 'runtime_report_sha256')}
                                 for language, entry in plan['candidates'].items()},
                     verifier_sha256=digest(__file__),
                     metrics_sha256=digest(Path(__file__).with_name('reserved_metrics_v9.py')),
                     selection_frozen_before_this_evaluation=True, analyst_blind=False)
    (args.output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    probes = {}
    for name, (filename, expected) in PROBES.items():
        path = args.probe_root/filename
        if digest(path) != expected:
            raise ValueError('Reserved probe changed')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len({r['case_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate case identity within a reserved set')
        probes[name] = rows
    results = []
    for language in ('ro', 'uk'):
        entry = plan['candidates'][language]
        model = discrete.DiscreteFieldsModel(native.NativeFieldsModel(entry['artifact'], entry['native_artifact'], entry['checkpoint'],
                                        entry['base'], entry['manifest'],
                                        expected_heads_sha256=entry['artifact_sha256'], device='cpu'))
        for name, all_rows in probes.items():
            rows = [r for r in all_rows if r['language'] == language]
            for offset in range(0, len(rows), 128):
                batch = rows[offset:offset+128]
                values = model.predict_many([dict(language=language, turns=r['turns']) for r in batch],
                                            focuses=[r.get('focus', len(r['turns'])-1) for r in batch])
                for row, value in zip(batch, values):
                    gold = row.get('review')
                    if gold is None:
                        gold = int(disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE')
                    if value['input_digest'] != input_key(row):
                        raise ValueError('Runtime input identity differs')
                    results.append(dict(probe=name, case_id=row['case_id'], language=language, gold=gold, **value))
        del model
    metrics = {name: {language: counts([r for r in results if r['probe'] == name and r['language'] == language])
                      for language in ('ro', 'uk')} for name in probes}
    unique = unique_records(results)
    report = dict(format='roguard-reserved-discrete-fields-v1', metrics=metrics,
                  unique_input_counts={language: counts([r for r in unique if r['language'] == language]) for language in ('ro', 'uk')},
                  total_rows=len(results), unique_inputs=len(unique),
                  passes=all(group['passes'] for dataset in metrics.values() for group in dataset.values()),
                  synthetic_only=True, independently_annotated=False, analyst_blind=False,
                  independent_observations=False, stable_release_ready=False,
                  limitation='Small, related authored probes; passing counts do not establish population accuracy or product readiness.',
                  selection_sha256=digest(args.output/'selection.json'),
                  probe_sha256={name: sha for name, (_, sha) in PROBES.items()})
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
