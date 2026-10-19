"""Evaluate frozen entailment reasons with the expanded sparse scope gate."""
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

from training.budgeted_native_frontier_v9 import frontier
from training.diagnose_contextual_complementarity_v9 import read_rows
from training.portable_scope_gate_v9 import ScopeGate
from training.prepare_instruction_sft_v8 import target


def relative_or(support, revision, limits):
    values = (support, revision, limits['support'], limits['revision'])
    if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid entailment score or cutoff')
    if min(limits.values()) <= 0:
        raise ValueError('Positive reason cutoffs required')
    ratio = max(support/limits['support'], revision/limits['revision'])
    return ratio/(1+ratio)


def metrics(predictions):
    result = {}
    for group in sorted({r['group'] for r in predictions}):
        rows = [r for r in predictions if r['group'] == group]
        positive = sum(r['gold'] for r in rows)
        negative = len(rows)-positive
        if not positive or not negative:
            raise ValueError('Both classes required in every group')
        missed = sum(r['gold'] and not r['predicted'] for r in rows)
        false = sum(not r['gold'] and r['predicted'] for r in rows)
        result[group] = dict(positive=positive, negative=negative, missed=missed, false_reviews=false,
            recall=(positive-missed)/positive, false_review_rate=false/negative,
            passes=missed*100 <= positive and false*20 <= negative)
    return result


def main():
    root = Path('research/v9-stabilization')
    study = root/'scope-entailment-153'
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen evidence differs: ' + path)
    protocol = json.loads((study/'protocol.json').read_text())
    if version('numpy') != protocol['numpy_version']:
        raise ValueError('Numeric dependency version differs')
    parent = root/'reason-coverage-123/candidate'
    parent_report = json.loads((parent/'development.json').read_text())
    if not parent_report.get('complete'):
        raise ValueError('Entailment training must be complete')
    entailment = json.loads((parent/'review.json').read_text())
    output = study/'actual'
    output.mkdir(exist_ok=False)
    report = dict(complete=False, languages={}, new_foundation_inference=False,
        fresh_test_inferred=False, synthetic_only=True, independently_annotated=False,
        stable_release_ready=False, calibration_used_for_selection=False,
        source_freeze_sha256=digest(study/'source-freeze.json'))
    for language in ('ro', 'uk'):
        gate_path = root/'expanded-scope-gate-147/actual'/(language+'-scope-gate.json.gz')
        gate = ScopeGate(json.loads(gzip.decompress(gate_path.read_bytes())))
        limits = entailment['thresholds'][language]
        def load(populations):
            rows = []
            for population in populations:
                inputs = read_rows(parent/(language+'_'+population+'.jsonl.gz'))
                lookup = {r['case_id']: r for r in read_rows(parent/(language+'_'+population+'-predictions.jsonl'))}
                if set(lookup) != {r['case_id'] for r in inputs}:
                    raise ValueError('Entailment population differs')
                for row in inputs:
                    row = dict(row, focus=row.get('focus', len(row['turns'])-1))
                    y = int(row['review']) if 'review' in row else target(row)
                    prediction = lookup[row['case_id']]
                    if y != prediction['gold']:
                        raise ValueError('Entailment label differs')
                    group = ('conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal') if population in ('dev', 'cal') else population
                    raw = max(prediction['support_score'], prediction['revision_score'])
                    margin = relative_or(prediction['support_score'], prediction['revision_score'], limits)
                    if int(margin >= .5) != prediction['predicted']:
                        raise ValueError('Normalized OR margin differs from frozen reason decision')
                    rows.append(dict(case_id=row['case_id'], language=language, group=group, gold=y,
                                     raw_max=raw, relative_or=margin, scope=gate.probability(row)))
            return rows
        dev = load(('dev', 'scope_dev')+(('stress',) if language == 'uk' else ()))
        best = selected = predictions = None
        comparisons = []
        for mode in protocol['reason_scores']:
            for power in protocol['scope_powers']:
                population = [dict(gold=r['gold'], group=r['group'], score=r[mode]*r['scope']**power) for r in dev]
                result = frontier(population)
                entry = dict(mode=mode, power=power, frontier=result)
                comparisons.append(entry)
                if best is None or tuple(result['rank']) < best:
                    best = tuple(result['rank'])
                    selected = entry
                    predictions = [dict(case_id=r['case_id'], language=language, **v,
                        predicted=int(v['score'] >= result['threshold'])) for r, v in zip(dev, population)]
        selection = dict(selected=selected, gate_sha256=digest(gate_path),
            entailment_artifact_sha256=digest(parent/'review.json'),
            entailment_checkpoint_sha256=entailment['checkpoint_sha256'], original_reason_cutoffs=limits)
        (output/(language+'-selection.json')).write_text(json.dumps(selection, indent=2)+'\n')
        (output/(language+'-dev-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        result = dict(selected=selected, comparisons=comparisons, calibration_scored=False,
                      selected_sha256=digest(output/(language+'-selection.json')))
        # Freeze selection before any calibration scoring, and skip failing languages.
        if selected['frontier']['counts_pass']:
            prior = load(('cal', 'exposed'))
            values = [r[selected['mode']]*r['scope']**selected['power'] for r in prior]
            predictions = [dict(case_id=r['case_id'], language=language, group=r['group'], gold=r['gold'],
                score=v, predicted=int(v >= selected['frontier']['threshold'])) for r, v in zip(prior, values)]
            (output/(language+'-prior-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
            result.update(calibration_scored=True, prior_metrics=metrics(predictions))
            result['prior_counts_pass'] = all(r['passes'] for r in result['prior_metrics'].values())
        report['languages'][language] = result
        print(language, json.dumps({k: v for k, v in result.items() if k != 'comparisons'}), flush=True)
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
        development_passes=all(r['selected']['frontier']['counts_pass'] for r in report['languages'].values()),
        prior_counts_pass=all(r.get('prior_counts_pass', False) for r in report['languages'].values()))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
