"""Check a fixed native/entailment-scope agreement candidate without retuning."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.fit_scope_entailment_v9 import metrics, relative_or
from training.portable_scope_gate_v9 import ScopeGate
from training.prepare_instruction_sft_v8 import target


def agreement_score(native, reason, native_cutoff, reason_cutoff):
    values = (native, reason, native_cutoff, reason_cutoff)
    if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid component score or cutoff')
    if min(native_cutoff, reason_cutoff) <= 0:
        raise ValueError('Positive component cutoffs required')
    ratio = min(native/native_cutoff, reason/reason_cutoff)
    return ratio/(1+ratio)


def main():
    root = Path('research/v9-stabilization')
    study = root/'scope-agreement-154'
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen evidence differs: ' + path)
    protocol = json.loads((study/'protocol.json').read_text())
    output = study/'actual'
    output.mkdir(exist_ok=False)
    report = dict(complete=False, languages={}, frozen_components=True, threshold_search=False,
                  fresh_test_inferred=False, new_foundation_inference=False,
                  calibration_used_for_selection=False, synthetic_only=True,
                  independently_annotated=False, stable_release_ready=False,
                  source_freeze_sha256=digest(study/'source-freeze.json'))
    reason_root = root/'reason-coverage-123/candidate'
    reason_spec = json.loads((reason_root/'review.json').read_text())
    for language in ('ro', 'uk'):
        native_root = root/'native-scope-operators-131'/language
        native_spec = json.loads(gzip.decompress((native_root/'review.json.gz').read_bytes()))
        native_cutoff = native_spec['thresholds'][language]
        selected = json.loads((root/'scope-entailment-153/actual'/(language+'-selection.json')).read_text())
        scope_selection = selected['selected']
        reason_cutoff = scope_selection['frontier']['threshold']
        gate_path = root/'expanded-scope-gate-147/actual'/(language+'-scope-gate.json.gz')
        if digest(gate_path) != selected['gate_sha256'] or digest(reason_root/'review.json') != selected['entailment_artifact_sha256']:
            raise ValueError('Scope or reason component identity differs')
        gate = ScopeGate(json.loads(gzip.decompress(gate_path.read_bytes())))
        specification = dict(format='roguard-scope-agreement-experiment-v1', language=language,
            native_artifact_sha256=digest(native_root/'review.json.gz'), native_checkpoint_sha256=native_spec['checkpoint_sha256'],
            entailment_artifact_sha256=digest(reason_root/'review.json'), entailment_checkpoint_sha256=reason_spec['checkpoint_sha256'],
            scope_artifact_sha256=digest(gate_path), native_cutoff=native_cutoff, reason_cutoff=reason_cutoff,
            reason_mode=scope_selection['mode'], scope_power=scope_selection['power'],
            original_reason_cutoffs=reason_spec['thresholds'][language], threshold=.5)
        # Both component cutoffs and their conjunction are fixed before calibration.
        (output/(language+'-composition.json')).write_text(json.dumps(specification, indent=2)+'\n')
        def score(populations):
            results = []
            for population in populations:
                inputs = read_rows(native_root/(population+'.jsonl.gz'))
                native = {r['case_id']: r for r in read_rows(native_root/(population+'-predictions.jsonl'))}
                reason = {r['case_id']: r for r in read_rows(reason_root/(language+'_'+population+'-predictions.jsonl'))}
                reason_inputs = {r['case_id']: r for r in read_rows(reason_root/(language+'_'+population+'.jsonl.gz'))}
                if set(native) != set(reason) or set(native) != {r['case_id'] for r in inputs}:
                    raise ValueError('Component populations differ')
                for raw in inputs:
                    row = dict(raw, focus=raw.get('focus', len(raw['turns'])-1))
                    key = row['case_id']
                    other = reason_inputs[key]
                    if row['turns'] != other['turns'] or row['focus'] != other.get('focus', len(other['turns'])-1):
                        raise ValueError('Component conversation content differs')
                    gold = int(row['review']) if 'review' in row else target(row)
                    if native[key]['gold'] != gold or reason[key]['gold'] != gold:
                        raise ValueError('Component gold labels differ')
                    reasons = reason[key]
                    base = max(reasons['support_score'], reasons['revision_score']) if scope_selection['mode'] == 'raw_max' else relative_or(reasons['support_score'], reasons['revision_score'], specification['original_reason_cutoffs'])
                    gated = base*gate.probability(row)**scope_selection['power']
                    value = agreement_score(native[key]['score'], gated, native_cutoff, reason_cutoff)
                    expected = native[key]['score'] >= native_cutoff and gated >= reason_cutoff
                    if int(value >= .5) != int(expected):
                        raise ValueError('Numeric agreement differs from conjunction')
                    group = ('conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal') if population in ('dev', 'cal') else population
                    results.append(dict(case_id=key, language=language, group=group, gold=gold,
                                        score=value, predicted=int(value >= .5)))
            return results
        dev = score(('dev', 'scope_dev')+(('stress',) if language == 'uk' else ()))
        development = metrics(dev)
        (output/(language+'-dev-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in dev))
        result = dict(development=development, development_passes=all(m['passes'] for m in development.values()),
                      composition_sha256=digest(output/(language+'-composition.json')), calibration_scored=False)
        if result['development_passes']:
            prior = score(('cal', 'exposed'))
            (output/(language+'-prior-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in prior))
            result.update(calibration_scored=True, prior_metrics=metrics(prior))
            result['prior_counts_pass'] = all(m['passes'] for m in result['prior_metrics'].values())
        report['languages'][language] = result
        print(language, json.dumps(result), flush=True)
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
        prior_counts_pass=all(r.get('prior_counts_pass', False) for r in report['languages'].values()))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
