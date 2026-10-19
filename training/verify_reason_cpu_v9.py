"""Full CPU replay of an already-qualified entailment candidate.

The earlier study124 CPU sample is a parity smoke check. This checks every
development, calibration, exposed regression and already-opened fresh case on
the default CPU runtime. It does not claim another first-use evaluation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from training.train_bare_distress_v9 import digest, read
from training.verify_reason_entailment_v9 import qualified
from training.prepare_instruction_sft_v8 import target
from training.verify_candidate_pipeline_v9 import counts


def ready(parent, installed):
    if not qualified(parent):
        return False
    if any(installed.get(k) is not True for k in ('complete','parity_pass','prerequisite_accuracy_pass',
                                                 'fresh_test_inferred','fresh_counts_pass')):
        return False
    if set(installed.get('fresh', {})) != {'ro','uk'}:
        return False
    for language in ('ro','uk'):
        c = installed['fresh'][language].get('counts', {})
        values = [c.get(k) for k in ('positive','negative','missed','false_reviews','abstentions')]
        if any(type(v) is not int for v in values):
            return False
        pos, neg, missed, false, abstain = values
        if pos != 150 or neg != 150 or not 0 <= missed <= pos or not 0 <= false <= neg or abstain:
            return False
        if missed * 100 > pos or false * 20 > neg:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--installed-root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    args = parser.parse_args()
    parent = Path('research/v9-stabilization/reason-coverage-123/candidate')
    study = Path('research/v9-stabilization/reason-entailment-installed-124')
    prior = study/'actual'
    record = json.loads((parent/'development.json').read_text())
    installed = json.loads((prior/'report.json').read_text())
    if not ready(record, installed):
        raise ValueError('Prior accuracy, installed parity and first-use fresh evaluation must pass')
    import roguard.reason_entailment as runtime
    import torch
    torch.set_num_threads(4)
    if args.installed_root.resolve() not in Path(runtime.__file__).resolve().parents:
        raise ValueError('Not the isolated installed runtime')
    frozen = json.loads((study/'source-freeze.json').read_text())
    for name, sha in frozen.items():
        if digest(Path(name)) != sha:
            raise ValueError('Frozen source changed: '+name)
        if name.startswith('src/roguard/') and digest(args.installed_root/name.removeprefix('src/')) != sha:
            raise ValueError('Installed source differs: '+name)
    inputs = {}
    for split in ('ro_dev','uk_dev','ro_scope_dev','uk_scope_dev','uk_stress',
                  'ro_cal','uk_cal','ro_exposed','uk_exposed'):
        inputs[split] = (parent/(split+'.jsonl.gz'), parent/(split+'-predictions.jsonl'))
    for language in ('ro','uk'):
        inputs[language+'_test'] = (Path('research/v9-stabilization/paired-scope-117/data')/(language+'-test.jsonl'),
                                   prior/(language+'-test-predictions.jsonl'))
    output = study/'cpu-full'
    output.mkdir(exist_ok=False)
    binding = dict(started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=digest(Path(__file__)),
                   runtime_source_freeze_sha256=digest(study/'source-freeze.json'),
                   protocol_sha256=digest(study/'cpu-full-protocol.json'),
                   wheel_sha256=digest(args.wheel), artifact_sha256=digest(parent/'review.json'),
                   training_report_sha256=digest(parent/'development.json'), installed_report_sha256=digest(prior/'report.json'),
                   first_use_evaluation=False,
                   input_sha256={str(p):digest(p) for pair in inputs.values() for p in pair})
    if binding['artifact_sha256'] != record['artifact_sha256']:
        raise ValueError('Candidate artifact identity changed')
    (output/'selection.json').write_text(json.dumps(binding, indent=2)+'\n')
    model = runtime.ReasonEntailmentReview(parent/'review.json', parent/'selected-model.safetensors', args.base,
                                         expected_sha256=record['artifact_sha256'], device='cpu')
    results = {}
    for split, (input_path, prediction_path) in inputs.items():
        rows = read(input_path) if input_path.suffix == '.gz' else [json.loads(line) for line in input_path.read_text().splitlines()]
        reference = [json.loads(line) for line in prediction_path.read_text().splitlines()]
        if len(rows) != len(reference) or [r['case_id'] for r in rows] != [r['case_id'] for r in reference]:
            raise ValueError('Input/reference coverage differs')
        values = []
        start = time.monotonic()
        for offset in range(0, len(rows), 64):
            batch = rows[offset:offset+64]
            outputs = model.predict_many([dict(language=r['language'], turns=r['turns']) for r in batch],
                                         focuses=[r.get('focus',len(r['turns'])-1) for r in batch])
            for row, value, old in zip(batch, outputs, reference[offset:offset+64]):
                gold = int(row['review']) if 'review' in row else target(row)
                if old['gold'] != gold:
                    raise ValueError('Reference gold differs')
                values.append(dict(value, case_id=row['case_id'], gold=gold))
            print(split, len(values), '/', len(rows), flush=True)
        if len(values) != len(reference) or any(v['score'] is None for v in values):
            raise ValueError('Incomplete CPU inference')
        changes = sum(v['predicted'] != old['predicted'] for v, old in zip(values, reference))
        maximum = max(abs(v[k]-old[k]) for v, old in zip(values, reference) for k in ('score','support_score','revision_score'))
        groups = ({kind:counts([v for row,v in zip(rows,values) if (row.get('source')=='v7_training_frame_projection')==flag])
                   for kind,flag in (('formal',False),('conversational',True))}
                  if split in ('ro_dev','uk_dev','ro_cal','uk_cal') else {'all':counts(values)})
        results[split] = dict(rows=len(values), decision_changes=changes, maximum_score_error=maximum,
                              parity_pass=changes==0 and maximum<=.001, counts=groups,
                              seconds=time.monotonic()-start)
        (output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
    report = dict(complete=True, splits=results, parity_pass=all(r['parity_pass'] for r in results.values()),
                  accuracy_pass=all(c['passes'] for r in results.values() for c in r['counts'].values()),
                  selection_sha256=digest(output/'selection.json'), first_use_evaluation=False,
                  synthetic_only=True, independently_annotated=False, stable_release_ready=False,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
