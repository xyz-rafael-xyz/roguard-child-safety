"""Reject promotion when any required evidence is missing or an error remains."""
from __future__ import annotations

from .trajectory_evaluation import evaluate, paired_compare
from .trajectory_model import RESPONSE_LABELS



def _response_assessment(rows):
    """Require valid distinct results spanning both languages and every label."""
    required = {lang+':'+label for lang in ('ro', 'uk') for label in RESPONSE_LABELS}
    groups = {}; seen = set(); errors = 0; invalid = 0; failures = []
    if rows is None or (isinstance(rows, list) and not rows):
        return {'rows': 0, 'errors': 0, 'invalid_rows': 0, 'groups': {},
                'missing_groups': sorted(required)}, ['RESPONSE_EVALUATION_MISSING']
    if not isinstance(rows, list):
        return {'rows': None, 'errors': None, 'invalid_rows': None, 'groups': {},
                'missing_groups': sorted(required)}, ['RESPONSE_EVALUATION_INVALID_ROWS']
    for row in rows:
        if not isinstance(row, dict):
            invalid += 1; errors += 1; continue
        identifier = row.get('case_id'); lang = row.get('language'); gold = row.get('gold')
        prediction = row.get('prediction')
        valid = (isinstance(identifier, str) and bool(identifier.strip()) and identifier not in seen
                 and isinstance(lang, str) and lang in ('ro', 'uk')
                 and isinstance(gold, str) and gold in RESPONSE_LABELS
                 and isinstance(prediction, dict) and isinstance(prediction.get('label'), str)
                 and prediction['label'] in RESPONSE_LABELS
                 and prediction.get('valid', True) is True)
        if not valid:
            invalid += 1; errors += 1; continue
        seen.add(identifier)
        group = groups.setdefault(lang+':'+gold, {'rows': 0, 'errors': 0})
        wrong = prediction['label'] != gold
        group['rows'] += 1; group['errors'] += wrong; errors += wrong
    missing = sorted(required-set(groups))
    if invalid: failures.append('RESPONSE_EVALUATION_INVALID_ROWS')
    if errors: failures.append('RESPONSE_EVALUATION_ERRORS')
    if missing: failures.append('RESPONSE_LANGUAGE_CATEGORY_COVERAGE_MISSING')
    return {'rows': len(rows), 'errors': errors, 'invalid_rows': invalid,
            'groups': groups, 'missing_groups': missing}, failures


def assess(cases, predictions, *, baseline=None, regression_cases=None,
           regression_predictions=None, response_rows=None):
    report=evaluate(cases,predictions)
    failures=[]
    if set(report['groups'])!={'ro','uk'}:failures.append('BILINGUAL_EVALUATION_MISSING')
    for lang,group in report['groups'].items():
        if group['strict_accuracy']!=1:failures.append(lang+':BINARY_ERROR_OR_ABSTENTION')
        if group['joint_set_coverage']!=1:failures.append(lang+':JOINT_COVERAGE_NOT_COMPLETE')
        if group['mean_set_size'] is None or group['mean_set_size']>=432:failures.append(lang+':SETS_MISSING_OR_VACUOUS')
        if group['annotated_evidence_retention']!=1:failures.append(lang+':EVIDENCE_RETENTION_NOT_COMPLETE')
        if not group['positive_checkpoints'] or not group['negative_checkpoints']:failures.append(lang+':CLASS_COVERAGE_MISSING')
    comparison=None
    if baseline is None:failures.append('MATCHED_BASELINE_MISSING')
    else:
        comparison=paired_compare(cases,baseline,predictions)
        baseline_report=evaluate(cases,baseline)
        retention_gain=set(baseline_report['groups'])==set(report['groups']) and all(
            g['annotated_evidence_retention'] is not None
            and baseline_report['groups'][lang]['annotated_evidence_retention'] is not None
            and g['annotated_evidence_retention']>baseline_report['groups'][lang]['annotated_evidence_retention']
            for lang,g in report['groups'].items())
        comparison['superiority_basis']='binary_accuracy' if comparison['strict_accuracy_delta']>0 else (
            'annotated_evidence_retention_at_equal_binary_accuracy' if comparison['strict_accuracy_delta']==0 and retention_gain else None)
        if comparison['superiority_basis'] is None or comparison['baseline_only_correct']:
            failures.append('STRICT_PAIRED_SUPERIORITY_NOT_SHOWN')
    regression=None
    if regression_cases is None or regression_predictions is None:failures.append('V9_REGRESSION_MISSING')
    else:
        regression=evaluate(regression_cases,regression_predictions)
        if set(regression['groups'])!={'ro','uk'} or any(g['strict_accuracy']!=1 for g in regression['groups'].values()):
            failures.append('V9_REGRESSION_ERRORS')
        if any(g['joint_set_coverage']!=1 for g in regression['groups'].values()):
            failures.append('V9_REGRESSION_JOINT_COVERAGE_NOT_COMPLETE')
    response_report, response_failures = _response_assessment(response_rows)
    failures.extend(response_failures)
    # A synthetic engineering run cannot satisfy an independent final study.
    external=['INDEPENDENT_SEALED_EVALUATION_REQUIRED','MEASURED_REVIEWER_BENEFIT_REQUIRED',
              'LARGER_CONTEXT_COMPARATOR_REQUIRED','CROSS_PLATFORM_INSTALLED_RUNTIME_REQUIRED']
    return {'format':'roguard-v10-promotion-gate-1','engineering_failures':failures,
            'external_requirements':external,'engineering_pass':not failures,
            'promotion_pass':False,'stable_release_ready':False,'comparison':comparison,
            'evaluation':report,'regression':regression,'response_evaluation':response_report,
            'decision':'REJECT','reason':'All required evidence must pass; no override from a perfect aggregate score.'}
