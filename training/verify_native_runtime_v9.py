"""Replay a completed native candidate through an independently installed wheel.

Run this file by absolute path from outside the repository with PYTHONPATH set
to a wheel installation directory. No training module is imported.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(rows, predictions):
    result = {}
    for kind in ('formal', 'conversational'):
        members = [(row, prediction) for row, prediction in zip(rows, predictions)
                   if ('conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal') == kind]
        positive = sum(p['gold'] for _, p in members)
        negative = len(members)-positive
        missed = sum(p['gold'] == 1 and p['predicted'] != 1 for _, p in members)
        false = sum(p['gold'] == 0 and p['predicted'] != 0 for _, p in members)
        result[kind] = dict(positive=positive, negative=negative, missed=missed, false_reviews=false,
                            recall=1-missed/positive if positive else None,
                            false_review_rate=false/negative if negative else None,
                            passes=bool(positive and negative and missed*100 <= positive and false*20 <= negative))
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('candidate', 'base', 'manifest', 'installed-root', 'wheel', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    import roguard
    import roguard.native_review as native
    from roguard.evidence import disposition
    import torch
    torch.set_num_threads(4)
    installed = a.installed_root.resolve()
    for module in (roguard, native):
        if installed not in Path(module.__file__).resolve().parents:
            raise ValueError('Runtime was not imported from the isolated wheel installation')
    development = json.loads((a.candidate/'development.json').read_text())
    artifact = a.candidate/'review.json.gz'
    if 'calibration_audit' not in development or development['selected_checkpoint_dev_replay_max_error'] > 1e-5:
        raise ValueError('Candidate training and checkpoint replay did not complete')
    model = native.NativeReviewModel(artifact, a.candidate/'selected-model.safetensors', a.base, a.manifest,
                                    expected_artifact_sha256=development['artifact_sha256'], device='cpu')
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(format='roguard-installed-native-replay-v1', device='cpu', synthetic_only=True,
                  installed_version=importlib.metadata.version('roguard-child-safety'),
                  installed_module=str(Path(native.__file__).resolve()), runtime_source_sha256=digest(Path(native.__file__)),
                  wheel_sha256=digest(a.wheel), artifact_sha256=digest(artifact), splits={},
                  maximum_score_tolerance=1e-4, permitted_decision_disagreements=0,
                  reserved_probes_used=False, stable_release_ready=False)
    for split in ('dev', 'cal'):
        data = a.candidate/(split+'.jsonl.gz')
        rows = [json.loads(line) for line in gzip.decompress(data.read_bytes()).splitlines()]
        reference_path = a.candidate/(split+'-predictions.jsonl')
        reference = [json.loads(line) for line in reference_path.read_text().splitlines()]
        ids = [r['case_id'] for r in rows]
        if len(set(ids)) != len(ids) or ids != [r['case_id'] for r in reference]:
            raise ValueError('Reference coverage/order differs')
        results = []
        start = time.monotonic()
        for offset in range(0, len(rows), 128):
            batch = rows[offset:offset+128]
            values = model.predict_many([dict(language=r['language'], turns=r['turns']) for r in batch],
                                        focuses=[r['focus'] for r in batch])
            for row, value in zip(batch, values):
                gold = int(disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE')
                results.append(dict(case_id=row['case_id'], gold=gold, **value))
            print(split, len(results), 'of', len(rows), flush=True)
        if any(r['gold'] != old['gold'] for r, old in zip(results, reference)):
            raise ValueError('Installed disposition policy differs from training labels')
        abstentions = sum(r['score'] is None for r in results)
        max_error = max((abs(r['score']-old['score']) for r, old in zip(results, reference) if r['score'] is not None), default=None)
        disagreements = sum(r['predicted'] != old['predicted'] for r, old in zip(results, reference))
        counts = audit(rows, results)
        report['splits'][split] = dict(rows=len(rows), input_sha256=digest(data), reference_sha256=digest(reference_path),
            seconds=time.monotonic()-start, abstentions=abstentions, max_score_error=max_error,
            decision_disagreements=disagreements, counts=counts,
            parity_pass=bool(not abstentions and max_error is not None and max_error <= 1e-4 and disagreements == 0),
            accuracy_gates_pass=all(c['passes'] for c in counts.values()))
        (a.output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in results))
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    report['complete'] = True
    report['parity_pass'] = all(s['parity_pass'] for s in report['splits'].values())
    report['accuracy_gates_pass'] = all(s['accuracy_gates_pass'] for s in report['splits'].values())
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not report['parity_pass']:
        raise SystemExit('Installed runtime parity failed')


if __name__ == '__main__':
    main()
