"""Compare a fixed categorical policy on research and actual installed fields."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

if __package__:
    from .reserved_metrics_v9 import counts, input_key
else:
    from reserved_metrics_v9 import counts, input_key


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    data = Path(path).read_bytes()
    if str(path).endswith('.gz'):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('research', 'installed'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--language', choices=('ro', 'uk'), required=True)
    parser.add_argument('--installed-root', type=Path)
    args = parser.parse_args()
    from roguard.evidence import ONTOLOGY, disposition
    import roguard.discrete_fields as runtime
    root = args.root
    study = root / 'discrete-policy-69' / args.language
    parent = root / 'scope-repair-66' / args.language
    data = root / 'scope-repair-66/encoders' / args.language
    frozen = json.loads((root / 'discrete-policy-69/source-freeze.json').read_text())
    if sha(runtime.__file__) != frozen['src/roguard/discrete_fields.py']:
        raise ValueError('Categorical runtime differs from declared comparison')
    selected = json.loads((parent / 'selection.json').read_text())
    heads_sha = sha(parent / 'selected-heads.json.gz')
    if selected['artifact_sha256'] != heads_sha:
        raise ValueError('Parent selection changed')
    if args.stage == 'research':
        study.mkdir(parents=True, exist_ok=False)
        selection = dict(created_utc=datetime.now(timezone.utc).isoformat(), rule=runtime.CONTRACT,
            contract_digest=runtime.CONTRACT_DIGEST, artifact_sha256=heads_sha,
            parent_selection_sha256=sha(parent / 'selection.json'),
            synthetic_only=True, calibration_used_for_selection=False,
            reserved_probes_used=False, stable_release_ready=False)
        (study / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
        output = study
        parent_report = None
    else:
        if args.installed_root is None or args.installed_root.resolve() not in Path(runtime.__file__).resolve().parents:
            raise ValueError('Categorical runtime must come from isolated wheel')
        selection = json.loads((study / 'selection.json').read_text())
        if (selection['artifact_sha256'] != heads_sha or selection['contract_digest'] != runtime.CONTRACT_DIGEST
                or selection['parent_selection_sha256'] != sha(parent / 'selection.json')):
            raise ValueError('Declared categorical selection changed')
        parent_path = root / 'installed-scope-repair-67' / args.language / 'report.json'
        parent_report = json.loads(parent_path.read_text())
        if (parent_report.get('complete') is not True or parent_report.get('artifact_sha256') != heads_sha
                or parent_report.get('selection_sha256') != sha(parent / 'selection.json')
                or parent_report.get('runtime_source_sha256') != frozen['src/roguard/native_fields.py']):
            raise ValueError('Actual installed parent evidence is incomplete or changed')
        output = study / 'installed'
        output.mkdir(exist_ok=False)
    report = dict(format='roguard-installed-discrete-fields-replay-v1' if parent_report else 'roguard-discrete-fields-research-v1',
        language=args.language, artifact_sha256=heads_sha, native_artifact_sha256=sha(data / 'review.json.gz'),
        selection_sha256=sha(study / 'selection.json'), runtime_source_sha256=sha(runtime.__file__),
        contract_digest=runtime.CONTRACT_DIGEST, splits={}, synthetic_only=True,
        reserved_probes_used=False, stable_release_ready=False)
    if parent_report:
        report.update(source_runtime_report_sha256=sha(parent_path), wheel_sha256=parent_report['wheel_sha256'],
                      installed_module=str(Path(runtime.__file__).resolve()), neural_outputs_reused_from_actual_installed_parent=True)
    for split in ('dev', 'cal'):
        rows = read(data / (split + '.jsonl.gz'))
        source = parent if not parent_report else root / 'installed-scope-repair-67' / args.language
        source_path = source / (split + '-predictions.jsonl')
        predictions = read(source_path)
        if [r['case_id'] for r in rows] != [r['case_id'] for r in predictions]:
            raise ValueError('Semantic input coverage differs')
        if parent_report and (parent_report['splits'][split]['predictions_sha256'] != sha(source_path)
                or parent_report['splits'][split]['input_sha256'] != sha(data / (split + '.jsonl.gz'))):
            raise ValueError('Installed neural outputs or inputs changed')
        results = []
        for row, value in zip(rows, predictions):
            gold = int(disposition({f: {'label': v} for f, v in row['attributes'].items()})[0] != 'CONTINUE')
            if value['gold'] != gold:
                raise ValueError('Policy labels differ')
            if parent_report:
                result = runtime.review(dict(language=args.language, turns=row['turns']), value,
                    expected_model_identity=heads_sha, focus=row['focus'])
            else:
                probabilities = value['field_probabilities']
                interpretation = {f: labels[max(range(len(labels)), key=probabilities[f].__getitem__)] for f, labels in ONTOLOGY.items()}
                result = dict(predicted=int(disposition({f: {'label': v} for f, v in interpretation.items()})[0] != 'CONTINUE'),
                    interpretation=interpretation, field_probabilities=probabilities, input_digest=input_key(row))
            results.append(dict(case_id=row['case_id'], gold=gold,
                kind='conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal', **result))
        metrics = {kind: counts([r for r in results if r['kind'] == kind]) for kind in ('formal', 'conversational')}
        observed = dict(rows=len(rows), counts=metrics, accuracy_gates_pass=all(v['passes'] for v in metrics.values()),
                        source_predictions_sha256=sha(source_path))
        if parent_report:
            reference = read(study / (split + '-predictions.jsonl'))
            changes = sum(a['predicted'] != b['predicted'] for a, b in zip(results, reference))
            labels = sum(a['interpretation'] != b['interpretation'] for a, b in zip(results, reference))
            abstentions = sum(r['predicted'] is None for r in results)
            error = max((abs(x-y) for a, b in zip(results, reference) if a['field_probabilities'] is not None
                for field in ONTOLOGY for x, y in zip(a['field_probabilities'][field], b['field_probabilities'][field])), default=None)
            observed.update(decision_changes=changes, interpretation_changes=labels, abstentions=abstentions,
                maximum_field_error=error, parity_pass=not (changes or labels or abstentions) and error is not None and error <= 1e-4)
        report['splits'][split] = observed
        (output / (split + '-predictions.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in results))
        print(json.dumps(dict(language=args.language, split=split, stage=args.stage, **observed)), flush=True)
    report.update(complete=True, accuracy_gates_pass=all(v['accuracy_gates_pass'] for v in report['splits'].values()))
    if parent_report:
        report['parity_pass'] = all(v['parity_pass'] for v in report['splits'].values())
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
