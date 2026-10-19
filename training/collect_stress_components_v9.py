"""Record actual frozen model evidence on already inspected development stress data."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'bundle', 'installed-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--bundle-sha256', required=True)
    a = p.parse_args()
    study = a.root/'research/v9-stabilization/stress-components-106'
    import roguard.model_bundle as runtime
    import torch
    torch.set_num_threads(4)
    from training.verify_candidate_pipeline_v9 import Capture
    installed = a.installed_root.resolve()
    if installed not in Path(runtime.__file__).resolve().parents:
        raise ValueError('Runtime must be imported from the isolated installation')
    for name, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if sha(a.root/name) != expected:
            raise ValueError('Frozen evidence differs: '+name)
        if name.startswith('src/roguard/') and sha(installed/name.removeprefix('src/')) != expected:
            raise ValueError('Installed runtime source differs')
    original = a.root/'research/v9-stabilization/installed-repaired-99'
    rows = [json.loads(l) for l in (original/'stress.jsonl').read_text().splitlines()]
    reference = [json.loads(l) for l in (original/'actual/stress-predictions.jsonl').read_text().splitlines()]
    if [r['case_id'] for r in rows] != [r['case_id'] for r in reference]:
        raise ValueError('Stress reference coverage differs')
    model = runtime.load_bundle(a.bundle, expected_sha256=a.bundle_sha256)
    if model.identity != sha(original/'fusion.json'):
        raise ValueError('Candidate differs from study99')
    captures = {}
    for name in ('native', 'semantic', 'interpreter'):
        captures[name] = Capture(getattr(model, name))
        setattr(model, name, captures[name])
    a.output.mkdir(parents=True, exist_ok=False)
    results = []; start = time.monotonic()
    for offset in range(0, len(rows), 20):
        batch = rows[offset:offset+20]
        outputs = model.predict_many([dict(language=r['language'], turns=r['turns']) for r in batch],
                                     focuses=[r.get('focus', len(r['turns'])-1) for r in batch])
        results.extend(outputs)
        print(len(results), 'of', len(rows), flush=True)
    error = max(abs(r['score']-old['score']) for r,old in zip(results,reference))
    changes = sum(r['predicted'] != old['predicted'] for r,old in zip(results,reference))
    if len(results) != len(rows) or error > .001 or changes:
        raise ValueError('Actual stress replay differs from study99')
    for name,capture in captures.items():
        if len(capture.values) != len(rows):
            raise ValueError('Incomplete component output')
        values = [dict(value, case_id=row['case_id'], gold=old['gold'])
                  for row,old,value in zip(rows,reference,capture.values)]
        (a.output/(name+'.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
    report = dict(complete=True, rows=len(rows), seconds=time.monotonic()-start,
        maximum_score_error=error, decision_changes=changes, bundle_sha256=a.bundle_sha256,
        input_sha256=sha(original/'stress.jsonl'), reference_sha256=sha(original/'actual/stress-predictions.jsonl'),
        source_freeze_sha256=sha(study/'source-freeze.json'),
        actual_models_loaded=True, cached_outputs_used=False,
        reserved_probes_used=False, synthetic_only=True, independent_accuracy_evaluation=False,
        stable_release_ready=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
