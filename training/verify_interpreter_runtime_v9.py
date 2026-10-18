"""Replay the frozen screen through an installed interpreter, without training imports."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('archive', 'base', 'installed-root', 'wheel', 'data', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--archive-sha256', required=True)
    a = p.parse_args()
    import roguard
    import roguard.interpreter as runtime
    from roguard.evidence import disposition
    from roguard.review_evaluation import evaluate
    import mlx.core as mx
    mx.set_cache_limit(512*1024*1024)
    for module in (roguard, runtime):
        if a.installed_root.resolve() not in Path(module.__file__).resolve().parents:
            raise ValueError('Runtime must be imported from the isolated wheel installation')
    rows = [json.loads(line) for line in a.data.read_text().splitlines()]
    reference = [json.loads(line) for line in a.reference.read_text().splitlines()]
    ids = [r['case_id'] for r in rows]
    if not rows or len(set(ids)) != len(ids) or ids != [r['case_id'] for r in reference]:
        raise ValueError('Reference coverage/order differs')
    model = runtime.InterpreterModel(a.archive, a.base, expected_archive_sha256=a.archive_sha256)
    a.output.mkdir(parents=True, exist_ok=False)
    outputs, expected = [], []
    start = time.monotonic()
    with (a.output/'predictions.jsonl').open('x') as stream:
        for i, row in enumerate(rows):
            value = model.predict(dict(language=row['language'], turns=row['turns']), focus=row['focus'])
            gold = int(disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE')
            outputs.append(dict(case_id=row['case_id'], gold=gold, **value))
            expected.append(dict(case_id=row['case_id'], language=row['language'], gold=gold,
                kind='conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal'))
            stream.write(json.dumps(outputs[-1])+'\n'); stream.flush()
            if (i+1)%25 == 0:
                print(i+1, 'of', len(rows), flush=True)
    if any(r['gold'] != old['gold'] for r, old in zip(outputs, reference)):
        raise ValueError('Installed policy labels differ from reference')
    decision_changes = sum(r['predicted'] != old['predicted'] for r, old in zip(outputs, reference))
    schema_changes = sum(r['valid_label_tokens'] != old['valid_label_tokens'] for r, old in zip(outputs, reference))
    interpretation_changes = sum(r['interpretation'] != old['interpretation'] for r, old in zip(outputs, reference))
    unexpected_abstentions = sum(r['reason'] == 'INTERPRETER_CONTEXT_LIMIT' for r in outputs)
    differences = [abs(value-other) for row, old in zip(outputs, reference)
        if row['conditional_field_probabilities'] is not None
        for field, values in row['conditional_field_probabilities'].items()
        for value, other in zip(values, old['conditional_field_probabilities'][field])]
    error = max(differences, default=None)
    parity = bool(not (decision_changes or schema_changes or interpretation_changes or unexpected_abstentions)
                  and error is not None and error <= .001)
    record = dict(format='roguard-installed-interpreter-replay-v1', synthetic_only=True,
        stable_release_ready=False, reserved_probes_used=False, rows=len(rows), seconds=time.monotonic()-start,
        installed_version=importlib.metadata.version('roguard-child-safety'),
        installed_module=str(Path(runtime.__file__).resolve()), runtime_source_sha256=digest(Path(runtime.__file__)),
        wheel_sha256=digest(a.wheel), archive_sha256=digest(a.archive), contract_digest=runtime.CONTRACT_DIGEST,
        data_sha256=digest(a.data), reference_sha256=digest(a.reference),
        decision_changes=decision_changes, schema_changes=schema_changes, interpretation_changes=interpretation_changes,
        unexpected_context_abstentions=unexpected_abstentions, maximum_probability_error=error,
        probability_tolerance=.001, parity_pass=parity, strict_evaluation=evaluate(expected, outputs),
        qualification='Runtime parity on the reused development screen, not independent accuracy validation.')
    (a.output/'report.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    if not parity:
        raise SystemExit('Installed interpreter parity failed')


if __name__ == '__main__':
    main()
