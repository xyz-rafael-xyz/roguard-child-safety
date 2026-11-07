"""Recompute a changed-attention candidate's exposed regression decision.

This is an engineering rejection check, not an independent release approval.
It verifies saved logits, threshold decisions and paired reports without fitting.
"""
import argparse
import json
from pathlib import Path

from roguard.gemma4_review import _score
from roguard.trajectory_data import file_sha, read_jsonl
from roguard.trajectory_evaluation import checkpoint_id, evaluate, paired_compare
from training.evaluate_semantic_readout_v10 import world_from_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--selected-study', type=Path, required=True)
    parser.add_argument('--reference-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite an assessment')
    protocol = json.loads((a.study/'protocol.json').read_text())
    report = json.loads((a.study/'report.json').read_text())
    if protocol['format'] != 'roguard-semantic-tail-regression-1' or not report['complete']:
        raise ValueError('Expected a complete changed-attention regression')
    for key, path in (
            ('selection_report_sha256', a.selected_study/'report.json'),
            ('original_reference_report_sha256', a.reference_study/'report.json')):
        if file_sha(path) != protocol[key]:
            raise ValueError('Parent study binding differs')
    specs = {lang: json.loads((a.selected_study/(lang+'-review.json')).read_text())
             for lang in ('ro', 'uk')}
    for lang, spec in specs.items():
        if spec['adapter_sha256'] != protocol['selected_adapters'][lang]:
            raise ValueError('Candidate adapter binding differs')
    paths = {'broad': Path('research/v10-evidence/broad-v9-regression/cases.jsonl'),
             **{name: Path('research/v10-evidence/data')/(name+'.jsonl')
                for name in ('dev', 'challenge', 'v9-regression')}}
    result = {'format': 'roguard-semantic-candidate-assessment-1',
              'source_sha256': file_sha(Path(__file__)),
              'regression_report_sha256': file_sha(a.study/'report.json'),
              'protocol_sha256': file_sha(a.study/'protocol.json'),
              'populations': {}, 'engineering_failures': [],
              'fitting_performed': False, 'independent_test': False,
              'stable_release_ready': False, 'decision': 'REJECT',
              'external_requirements': ['FRESH_CALIBRATION', 'INDEPENDENT_FINAL_EVALUATION',
                  'QUALIFIED_RESPONSE_CHECKING', 'MEASURED_REVIEWER_BENEFIT',
                  'INDEPENDENT_STRONG_BASELINES', 'TARGET_PLATFORM_RUNTIME_CHECKS']}
    maximum_score_difference = 0.
    for name, path in paths.items():
        candidate_path = a.study/(name+'-candidate.jsonl')
        if file_sha(path) != protocol['inputs'][name] or file_sha(candidate_path) != report['prediction_sha256'][name]:
            raise ValueError('Evaluation input or prediction binding differs')
        cases = read_jsonl(path)
        predictions = read_jsonl(candidate_path)
        baseline = read_jsonl(a.reference_study/(name+'-baseline.jsonl'))
        for prediction in predictions:
            spec = specs[prediction['language']]
            score = _score(prediction['field_logits'], spec)
            error = abs(score-prediction['score'])
            maximum_score_difference = max(maximum_score_difference, error)
            if error > 1e-7 or int(score >= spec['threshold']) != prediction['predicted']:
                raise ValueError('Saved logits and binary decision differ')
            if prediction['interpretation_set']['hypotheses'] != [world_from_logits(prediction['field_logits'])]:
                raise ValueError('Saved logits and semantic argmax differ')
            if prediction['qualified'] or prediction['permits_external_action'] or prediction['establishes_safety']:
                raise ValueError('Research prediction claims qualification or authority')
        measured = evaluate(cases, predictions)
        old = evaluate(cases, baseline)
        comparison = paired_compare(cases, baseline, predictions)
        expected = report['populations'][name]
        if measured != expected['candidate'] or old != expected['baseline'] or comparison != expected['comparison']:
            raise ValueError('Stored report differs from recomputed predictions')
        by_id = {row['checkpoint_id']: row for row in predictions}
        old_by_id = {row['checkpoint_id']: row for row in baseline}
        errors, fixed, introduced = [], [], []
        for case in cases:
            for point in case['checkpoints']:
                key = checkpoint_id(case, point)
                correct = by_id[key]['predicted'] == point['review']
                old_correct = old_by_id[key]['predicted'] == point['review']
                if not correct:
                    errors.append(key)
                if correct and not old_correct:
                    fixed.append(key)
                if old_correct and not correct:
                    introduced.append(key)
        result['populations'][name] = {
            'checkpoints': len(predictions), 'binary_errors': errors,
            'fixed_binary_errors': fixed, 'introduced_binary_errors': introduced,
            'paired_comparison': comparison,
            'uncalibrated_joint_argmax_errors': {
                lang: round(g['checkpoints']*(1-g['joint_set_coverage']))
                for lang, g in measured['groups'].items()}}
        if errors:
            result['engineering_failures'].append(name+':BINARY_ERRORS')
        if introduced:
            result['engineering_failures'].append(name+':NEW_BINARY_REGRESSIONS')
        if any(g['joint_set_coverage'] != 1 for g in measured['groups'].values()):
            result['engineering_failures'].append(name+':SEMANTIC_ARGMAX_ERRORS')
    result['maximum_saved_score_difference'] = maximum_score_difference
    result['reason'] = 'Exposed development gains do not override regression errors or missing independent evidence.'
    a.output.mkdir(parents=True)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
