"""Compare a genuinely installed adapter runtime with every frozen prior result."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def installed_files(wheel, root):
    """Bind installed package bytes to the tested wheel, excluding its metadata."""
    result = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith('roguard/') or name.endswith('/'):
                continue
            target = root/name
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('Wheel package path escapes installation')
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            if digest(target) != expected:
                raise ValueError('Installed package differs from wheel: ' + name)
            result[name] = expected
    if 'roguard/gemma4_review.py' not in result:
        raise ValueError('Wheel lacks the adapter runtime')
    for name, module in list(sys.modules.items()):
        if name == 'roguard' or name.startswith('roguard.'):
            path = getattr(module, '__file__', None)
            if path is None or not Path(path).resolve().is_relative_to(root.resolve()):
                raise ValueError('RoGuard was imported from outside the installation')
            relative = str(Path(path).resolve().relative_to(root.resolve()))
            if relative not in result or digest(path) != result[relative]:
                raise ValueError('Loaded RoGuard module is not bound to this wheel')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--installed', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--wheel-sha256', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=Path('research/v9-stabilization/grounding-tail-fit-177'))
    parser.add_argument('--prior', type=Path, default=Path('research/v9-stabilization/grounding-tail-prior-178'))
    parser.add_argument('--study', type=Path, default=Path('research/v9-stabilization/grounding-tail-installed-180'))
    args = parser.parse_args()
    if digest(args.wheel) != args.wheel_sha256:
        raise ValueError('Wheel identity differs')
    package = installed_files(args.wheel, args.installed)
    from roguard.entailment_bundle import load_bundle
    from roguard.native_review import prepare_input
    from training.diagnose_contextual_complementarity_v9 import read_rows
    from training.shared_tail_qualification_v9 import candidate_bindings, compare_replay
    from training.stage_grounding_tail_bundle_v9 import qualification
    from training.fit_scope_entailment_v9 import metrics
    # Ensure helper imports did not resolve the checkout's installed package.
    installed_files(args.wheel, args.installed)
    root = Path('research/v9-stabilization')
    specs, bindings = candidate_bindings(args.candidate)
    bindings.update(qualification(root, args.prior))
    staged = json.loads((args.bundle.parent/'staging-evidence.json').read_text())
    if staged.get('complete') is not True or staged['bundle_sha256'] != digest(args.bundle) or staged['parents'] != bindings:
        raise ValueError('Staged bundle does not belong to this qualified candidate')
    entries = read_rows(args.prior/'inputs.jsonl.gz')
    study = args.study
    study.mkdir(exist_ok=False)
    paths = [Path('training')/name for name in ('verify_grounding_tail_installed_v9.py', 'stage_grounding_tail_bundle_v9.py',
        'shared_tail_qualification_v9.py', 'fit_scope_entailment_v9.py', 'diagnose_contextual_complementarity_v9.py')]
    bindings.update({str(path): digest(path) for path in paths})
    (study/'source-freeze.json').write_text(json.dumps(bindings, indent=2)+'\n')
    protocol = dict(study=int(study.name.rsplit('-', 1)[1]), wheel_sha256=args.wheel_sha256, bundle_sha256=digest(args.bundle),
        installed_files=package, rows=len(entries), score_tolerance=1e-5, required_decision_changes=0,
        populations=['cal', 'exposed'], batch_size=4, source_freeze_sha256=digest(study/'source-freeze.json'),
        synthetic_only=True, independently_annotated=False, fresh_test_inferred=False, stable_release_ready=False)
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    output = study/'actual'
    output.mkdir()
    report = dict(complete=False, languages={}, wheel_sha256=args.wheel_sha256,
        bundle_sha256=digest(args.bundle), protocol_sha256=digest(study/'protocol.json'),
        source_freeze_sha256=digest(study/'source-freeze.json'), installed_package_verified=True,
        fresh_test_inferred=False, synthetic_only=True, independently_annotated=False, stable_release_ready=False)
    model = load_bundle(args.bundle, expected_sha256=digest(args.bundle))
    started = time.monotonic()
    for language in ('ro', 'uk'):
        chosen = [r for r in entries if r['row']['language'] == language]
        actual = []
        tick = time.monotonic()
        for start in range(0, len(chosen), protocol['batch_size']):
            batch = chosen[start:start+protocol['batch_size']]
            payloads = [dict(language=language, turns=r['row']['turns']) for r in batch]
            focuses = [r['row']['focus'] for r in batch]
            values = model.predict_many(payloads, focuses=focuses)
            if len(values) != len(batch):
                raise ValueError('Incomplete installed inference')
            for entry, payload, focus, value in zip(batch, payloads, focuses, values):
                _, expected_focus, expected_digest = prepare_input(payload, focus)
                if (value['focus'] != expected_focus or value['input_digest'] != expected_digest
                        or value['language'] != language or type(value['predicted']) is not int or value['predicted'] not in (0, 1)
                        or type(value['score']) not in (int, float) or not math.isfinite(value['score'])
                        or not 0 <= value['score'] <= 1 or value['predicted'] != int(value['score'] >= specs[language]['threshold'])
                        or any(value.get(k) is not False for k in ('text_in_report', 'establishes_safety', 'permits_notification', 'permits_external_action'))):
                    raise ValueError('Installed output binding or contract differs')
                actual.append(dict(case_id=entry['row']['case_id'], language=language, group=entry['group'],
                    gold=entry['gold'], score=value['score'], predicted=value['predicted']))
            if start % 100 == 0:
                print('INSTALLED', language, len(actual), '/', len(chosen), 'seconds', round(time.monotonic()-tick, 2), flush=True)
        (output/f'{language}-prior-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in actual))
        expected = read_rows(args.prior/'actual'/f'{language}-prior-predictions.jsonl')
        replay = compare_replay(actual, expected, specs[language]['threshold'], protocol['score_tolerance'])
        measured = metrics(actual)
        if not all(v['passes'] for v in measured.values()):
            raise ValueError('Installed model fails prior accuracy gates')
        report['languages'][language] = dict(replay=replay, metrics=measured, seconds_including_model_load=time.monotonic()-tick)
        (output/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        print('VERIFIED', language, json.dumps(report['languages'][language]), flush=True)
    if installed_files(args.wheel, args.installed) != package:
        raise ValueError('Installed package changed during qualification')
    report.update(complete=True, installed_execution_verified=True, prior_counts_pass=True,
        seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat())
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
