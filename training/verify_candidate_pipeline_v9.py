"""Verify a frozen qualified composite through an isolated installed package."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(values):
    positive = sum(v['gold'] for v in values)
    negative = len(values) - positive
    missed = sum(v['gold'] == 1 and v['predicted'] != 1 for v in values)
    false = sum(v['gold'] == 0 and v['predicted'] != 0 for v in values)
    return dict(rows=len(values), positive=positive, negative=negative, missed=missed,
        false_reviews=false, abstentions=sum(v['predicted'] is None for v in values),
        recall=1-missed/positive if positive else None,
        false_review_rate=false/negative if negative else None,
        passes=bool(positive and negative and missed*100 <= positive and false*20 <= negative))


def require_counts(groups):
    if set(groups) != {'formal', 'conversational'}:
        raise ValueError('Missing qualification groups')
    for group in groups.values():
        values = [group.get(k) for k in ('positive', 'negative', 'missed', 'false_reviews', 'abstentions')]
        if (any(type(v) is not int for v in values) or values[0] <= 0 or values[1] <= 0
                or not 0 <= values[2] <= values[0] or not 0 <= values[3] <= values[1]
                or values[4] != 0 or values[2]*100 > values[0] or values[3]*20 > values[1]):
            raise ValueError('Qualification counts fail')


class Capture:
    """Record actual component outputs without replacing the pipeline path."""
    def __init__(self, model):
        self.model = model
        self.values = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, payload, *, focus=None):
        value = self.model.predict(payload, focus=focus)
        self.values.append(value)
        return value

    def predict_many(self, payloads, *, focuses=None):
        values = self.model.predict_many(payloads, focuses=focuses)
        self.values.extend(values)
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'bundle', 'installed-root', 'wheel', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--selection-directory', required=True)
    args = parser.parse_args()
    study = args.root / 'research/v9-stabilization' / args.study
    import torch
    torch.set_num_threads(4)
    import roguard.model_bundle as runtime
    from roguard.native_review import prepare_input
    from roguard.evidence import disposition
    installed = args.installed_root.resolve()
    if installed not in Path(runtime.__file__).resolve().parents:
        raise ValueError('Not an isolated installed runtime')
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if sha(args.root/path) != expected:
            raise ValueError('Frozen source or evidence differs: '+path)
        if path.startswith('src/roguard/') and sha(installed/path.removeprefix('src/')) != expected:
            raise ValueError('Installed runtime source differs: '+path)
    selected = args.root/'research/v9-stabilization'/args.selection_directory/'selection.json'
    selection = json.loads(selected.read_text())
    audit = json.loads(selected.with_name('audit.json').read_text())
    for group in (selection['development'], selection['oof'], audit['calibration']):
        require_counts(group)
    if audit['counts_pass'] is not True or audit['selection_sha256'] != sha(selected):
        raise ValueError('Qualified selection differs')
    model = runtime.load_bundle(args.bundle, expected_sha256=args.bundle_sha256)
    if (model.identity != sha(study/'fusion.json')
            or model.fusion._spec['selection_sha256'] != sha(selected)
            or model.fusion._spec['model'] != selection['model']
            or model.fusion._spec['threshold'] != selection['threshold']):
        raise ValueError('Actual composite differs from selected candidate')
    captures = {}
    for name in ('native', 'semantic', 'interpreter'):
        captures[name] = Capture(getattr(model, name))
        setattr(model, name, captures[name])
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(format='roguard-installed-repaired-pipeline-v1', language='uk',
        actual_models_loaded=True, cached_component_outputs_used=False,
        bundle_sha256=args.bundle_sha256, artifact_sha256=model.identity,
        selection_sha256=sha(selected), qualification_audit_sha256=sha(selected.with_name('audit.json')),
        wheel_sha256=sha(args.wheel), source_freeze_sha256=sha(study/'source-freeze.json'),
        reserved_probes_used=False, stable_release_ready=False,
        synthetic_only=True, independent_accuracy_evaluation=False, splits={})
    for split in ('screen', 'stress'):
        data = study/(split+'.jsonl')
        rows = [json.loads(line) for line in data.read_text().splitlines()]
        if len({r['case_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate input identity')
        reference = None
        if split == 'screen':
            reference = [json.loads(line) for line in (study/'reference.jsonl').read_text().splitlines()]
            if [r['case_id'] for r in reference] != [r['case_id'] for r in rows]:
                raise ValueError('Incomplete or reordered reference')
        results = []
        for capture in captures.values():
            capture.values.clear()
        start = time.monotonic()
        for offset in range(0, len(rows), 20):
            batch = rows[offset:offset+20]
            payloads = [dict(language=r['language'], turns=r['turns']) for r in batch]
            focuses = [r.get('focus', len(r['turns'])-1) for r in batch]
            outputs = model.predict_many(payloads, focuses=focuses)
            if len(outputs) != len(batch):
                raise ValueError('Incomplete actual pipeline output')
            for row, payload, focus, value in zip(batch, payloads, focuses, outputs):
                if value['input_digest'] != prepare_input(payload, focus)[2]:
                    raise ValueError('Output input binding differs')
                gold = int(disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE')
                results.append(dict(case_id=row['case_id'], gold=gold,
                    kind='conversational' if row.get('source')=='v7_training_frame_projection' else 'formal', **value))
            print(split, len(results), 'of', len(rows), flush=True)
        values = dict(rows=len(rows), seconds=time.monotonic()-start,
                      input_sha256=sha(data), counts=counts(results))
        if reference is not None:
            if any(r['gold'] != old['gold'] for r, old in zip(results, reference)):
                raise ValueError('Gold label binding differs')
            error = max((abs(r['score']-old['score']) for r, old in zip(results, reference)
                         if r['score'] is not None and old['score'] is not None), default=None)
            changes = sum(r['predicted'] != old['predicted'] for r, old in zip(results, reference))
            values.update(maximum_score_error=error, decision_changes=changes,
                parity_pass=error is not None and error <= .001 and changes == 0 and values['counts']['abstentions'] == 0,
                reference_sha256=sha(study/'reference.jsonl'),
                groups={k: counts([r for r in results if r['kind']==k]) for k in ('formal', 'conversational')})
        report['splits'][split] = values
        for name, capture in captures.items():
            if len(capture.values) != len(rows):
                raise ValueError('Incomplete captured component output')
            component_rows = [dict(value, case_id=row['case_id'], gold=result['gold'])
                              for row, result, value in zip(rows, results, capture.values)]
            (args.output/(split+'-'+name+'.jsonl')).write_text(
                ''.join(json.dumps(v)+'\n' for v in component_rows))
        (args.output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in results))
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        if split == 'screen' and not values['parity_pass']:
            raise SystemExit('Actual installed pipeline parity failed before stress audit')
    report.update(complete=True, parity_pass=report['splits']['screen']['parity_pass'],
                  stress_pass=report['splits']['stress']['counts']['passes'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
