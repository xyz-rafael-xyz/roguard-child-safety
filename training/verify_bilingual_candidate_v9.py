"""Check installed bilingual dispatch, then conditionally open frozen probes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

if __package__:
    from .verify_repaired_pipeline_v9 import require_counts
    from .verify_reserved_native_v9 import PROBES
    from .reserved_metrics_v9 import counts, input_key, unique_records
else:
    from verify_repaired_pipeline_v9 import require_counts
    from verify_reserved_native_v9 import PROBES
    from reserved_metrics_v9 import counts, input_key, unique_records


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_candidate(language, selection, audit, actual, fusion, selection_sha, bundle_sha):
    if language == 'ro':
        if set(selection['development']) != {'oof', 'refit'}:
            raise ValueError('Romanian development evidence is incomplete')
        for group in selection['development'].values():
            require_counts(group)
    elif language == 'uk':
        require_counts(selection['development'])
        require_counts(selection['oof'])
    else:
        raise ValueError('Unsupported language candidate')
    require_counts(audit['calibration'])
    if audit.get('counts_pass') is not True or audit.get('selection_sha256') != selection_sha:
        raise ValueError('Qualification audit differs')
    if (fusion.get('language') != language or fusion.get('selection_sha256') != selection_sha
            or fusion.get('model') != selection.get('model')
            or fusion.get('threshold') != selection.get('threshold')):
        raise ValueError('Export differs from qualified selection')
    if (actual.get('actual_models_loaded') is not True
            or actual.get('cached_component_outputs_used') is not False
            or actual.get('parity_pass') is not True or actual.get('bundle_sha256') != bundle_sha):
        raise ValueError('Missing actual installed inference evidence')
    if language == 'ro':
        if (actual.get('rows') != 200 or actual.get('decision_changes') != 0
                or type(actual.get('maximum_score_error')) not in (int, float)
                or not 0 <= actual['maximum_score_error'] <= .001):
            raise ValueError('Romanian installed parity failed')
    else:
        if actual.get('complete') is not True or actual.get('selection_sha256') != selection_sha:
            raise ValueError('Ukrainian installed audit incomplete')
        screen = actual.get('splits', {}).get('screen', {})
        stress = actual.get('splits', {}).get('stress', {}).get('counts', {})
        if (screen.get('rows') != 200 or screen.get('parity_pass') is not True
                or screen.get('decision_changes') != 0
                or type(screen.get('maximum_score_error')) not in (int, float)
                or not 0 <= screen['maximum_score_error'] <= .001):
            raise ValueError('Ukrainian installed screen failed')
        values = [stress.get(k) for k in ('rows', 'positive', 'negative', 'missed', 'false_reviews', 'abstentions')]
        if (any(type(v) is not int for v in values) or values[:3] != [216, 44, 172]
                or values[3] != 0 or not 0 <= values[4] <= 8
                or not 0 <= values[5] <= values[4] or actual.get('stress_pass') is not True):
            raise ValueError('Retained stress gate failed')


def checked_json(root, entry):
    if not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}:
        raise ValueError('Invalid evidence reference')
    path = (root/entry['path']).resolve()
    if root.resolve() not in path.parents or sha(path) != entry['sha256']:
        raise ValueError('Evidence path or identity differs')
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'plan', 'bundle', 'installed-root', 'wheel', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    study = root/'research/v9-stabilization/bilingual-candidate-100'
    if sha(args.plan) != args.plan_sha256:
        raise ValueError('Candidate plan differs')
    plan = json.loads(args.plan.read_text())
    if set(plan['languages']) != {'ro', 'uk'} or plan['wheel_sha256'] != sha(args.wheel):
        raise ValueError('Both pinned language candidates and installed wheel required')
    for language, entry in plan['languages'].items():
        documents = {key: checked_json(root, entry[key]) for key in ('selection', 'audit', 'actual', 'fusion')}
        validate_candidate(language, **documents, selection_sha=entry['selection']['sha256'], bundle_sha=entry['bundle_sha256'])
        if language == 'uk' and documents['actual'].get('artifact_sha256') != entry['fusion']['sha256']:
            raise ValueError('Installed Ukrainian model identity differs')
    import torch
    torch.set_num_threads(4)
    import roguard.language_bundle as runtime
    if args.installed_root.resolve() not in Path(runtime.__file__).resolve().parents:
        raise ValueError('Not an isolated installed runtime')
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if sha(root/path) != expected:
            raise ValueError('Frozen source differs: '+path)
        if path.startswith('src/roguard/') and sha(args.installed_root/path.removeprefix('src/')) != expected:
            raise ValueError('Installed source differs: '+path)
    model = runtime.LanguageReviewModel(args.bundle, expected_sha256=plan['bundle_sha256'])
    for language, entry in plan['languages'].items():
        if model._entries[language]['sha256'] != entry['bundle_sha256']:
            raise ValueError('Language dispatcher selects a different bundle')
    args.output.mkdir(parents=True, exist_ok=False)
    # This screen is already exposed development/calibration, never reserved.
    screen = [json.loads(line) for line in (study/'screen.jsonl').read_text().splitlines()]
    reference = [json.loads(line) for line in (study/'reference.jsonl').read_text().splitlines()]
    if len(screen) != 40 or [r['case_id'] for r in screen] != [r['case_id'] for r in reference]:
        raise ValueError('Bilingual screen coverage differs')
    start = time.monotonic()
    results = model.predict_many([dict(language=r['language'], turns=r['turns']) for r in screen],
                                 focuses=[r.get('focus', len(r['turns'])-1) for r in screen])
    if len(results) != len(screen):
        raise ValueError('Incomplete bilingual integration output')
    errors = []; changes = 0
    for row, result, old in zip(screen, results, reference):
        if (result['input_digest'] != input_key(row) or result['score'] is None
                or result['model_identity'] != plan['languages'][row['language']]['fusion']['sha256']):
            raise ValueError('Bilingual integration binding or abstention differs')
        changes += result['predicted'] != old['predicted']
        errors.append(abs(result['score']-old['score']))
    integration = dict(rows=len(results), seconds=time.monotonic()-start,
        decision_changes=changes, maximum_score_error=max(errors),
        parity_pass=changes == 0 and max(errors) <= .001, actual_models_loaded=True,
        bundle_sha256=plan['bundle_sha256'], plan_sha256=args.plan_sha256,
        wheel_sha256=sha(args.wheel), reserved_probes_used=False)
    (args.output/'integration.json').write_text(json.dumps(integration, indent=2)+'\n')
    (args.output/'integration-predictions.jsonl').write_text(''.join(
        json.dumps(dict(case_id=r['case_id'], **value))+'\n' for r, value in zip(screen, results)))
    if not integration['parity_pass']:
        raise SystemExit('Bilingual integration failed; reserved inputs remain unopened')
    # Freeze passing candidate + integration evidence before reading probes.
    selection = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=args.plan_sha256, integration_sha256=sha(args.output/'integration.json'),
        source_freeze_sha256=sha(study/'source-freeze.json'),
        selection_frozen_before_this_evaluation=True, analyst_blind=False)
    (args.output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    probes = {}
    for name, (filename, expected) in PROBES.items():
        path = root/'research/v8-stabilization'/filename
        if sha(path) != expected:
            raise ValueError('Reserved input identity differs')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len({r['case_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate identity within reserved probe')
        probes[name] = rows
    from roguard.evidence import disposition
    predictions = []
    for language in ('uk', 'ro'):
        cases = [(name, row) for name, rows in probes.items() for row in rows if row['language'] == language]
        for offset in range(0, len(cases), 128):
            batch = cases[offset:offset+128]
            values = model.predict_many([dict(language=language, turns=r['turns']) for _, r in batch],
                                        focuses=[r.get('focus', len(r['turns'])-1) for _, r in batch])
            if len(values) != len(batch):
                raise ValueError('Incomplete reserved predictions')
            for (name, row), value in zip(batch, values):
                gold = row.get('review')
                if gold is None:
                    gold = int(disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE')
                if value['input_digest'] != input_key(row):
                    raise ValueError('Reserved input binding differs')
                predictions.append(dict(probe=name, case_id=row['case_id'], gold=gold, **value))
            print(language, 'reserved', min(offset+128, len(cases)), 'of', len(cases), flush=True)
    metrics = {name: {language: counts([r for r in predictions if r['probe']==name and r['language']==language])
                     for language in ('ro', 'uk')} for name in probes}
    unique = unique_records(predictions)
    deduplicated = {language: counts([r for r in unique if r['language']==language]) for language in ('ro', 'uk')}
    report = dict(format='roguard-bilingual-reserved-evaluation-v1', metrics=metrics,
        unique_input_counts=deduplicated, total_rows=len(predictions), unique_inputs=len(unique),
        passes=all(group['passes'] for dataset in metrics.values() for group in dataset.values())
            and all(group['passes'] for group in deduplicated.values()),
        synthetic_only=True, independently_annotated=False, analyst_blind=False,
        independent_observations=False, stable_release_ready=False,
        selection_sha256=sha(args.output/'selection.json'),
        probe_sha256={name: expected for name, (_, expected) in PROBES.items()},
        limitation='Small related authored probes; passing counts do not establish population accuracy.')
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
