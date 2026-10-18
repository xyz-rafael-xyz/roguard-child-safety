"""Verify installed planning against actual installed interpreter outputs."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('installed-root', 'data', 'runtime-report', 'predictions', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    args = p.parse_args()
    import roguard.interpreter_assurance as bridge
    import roguard.interpreter as runtime
    for module in (bridge, runtime):
        if args.installed_root.resolve() not in Path(module.__file__).resolve().parents:
            raise ValueError('Modules must come from the installed wheel')
    evidence = json.loads(args.runtime_report.read_text())
    if (evidence.get('parity_pass') is not True or evidence['runtime_source_sha256'] != sha(runtime.__file__)
            or evidence['data_sha256'] != sha(args.data)):
        raise ValueError('Installed interpreter evidence is incomplete or different')
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    if len(rows) != evidence['rows'] or [r['case_id'] for r in rows] != [r['case_id'] for r in predictions]:
        raise ValueError('Installed screen coverage differs')
    records = []
    for row, prediction in zip(rows, predictions):
        payload = dict(language=row['language'], turns=row['turns'])
        identities = [evidence['archive_sha256']]
        plan = bridge.plan_review(payload, [prediction], expected_model_identities=identities, focus=row['focus'])
        if not bridge.verify_review_plan(plan, payload, [prediction], expected_model_identities=identities, focus=row['focus']):
            raise ValueError('Bound plan verification failed')
        certificate = plan['certificate']
        if prediction['predicted'] is None:
            if not plan['includes_abstention'] or certificate['status'] != 'CLARIFICATION_REQUIRED':
                raise ValueError('Abstention was incorrectly resolved')
        else:
            actions = certificate['possible_actions']
            if len(actions) != 1 or int(actions[0] != 'CONTINUE') != prediction['predicted']:
                raise ValueError('Planner changed the interpreter policy proposal')
        if plan['permits_external_action'] or plan['permits_notification'] or certificate['permits_external_action']:
            raise ValueError('Planner granted external permission')
        records.append(dict(case_id=row['case_id'], certificate_digest=certificate['certificate_digest'],
                            source_report_digests=plan['source_report_digests'],
                            status=certificate['status'], includes_abstention=plan['includes_abstention']))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'bindings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    result = dict(format='roguard-installed-interpreter-assurance-check-v1', rows=len(records), passes=True,
                  input_sha256=sha(args.data), predictions_sha256=sha(args.predictions), runtime_report_sha256=sha(args.runtime_report),
                  installed_bridge_sha256=sha(bridge.__file__), wheel_sha256=evidence['wheel_sha256'],
                  unchanged_policy_proposals=True, every_binding_recomputed=True, external_permissions_granted=0,
                  synthetic_only=True, independent_accuracy_evaluation=False, stable_release_ready=False)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
