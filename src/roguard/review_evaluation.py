"""Evaluate review proposals without hiding abstentions or missing predictions.

The scores are model outputs, not probabilities of a child's safety. This module
never grants permissions or dispatches notifications.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def triage(score, *, low, high, language, valid=True):
    """Apply caller-calibrated thresholds; uncertainty requests clarification.

    The caller must evaluate thresholds on separate calibration data. This
    function supplies no default threshold or statistical coverage guarantee.
    """
    for value in (low, high):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Thresholds must be finite numbers')
    if not 0 <= low < high <= 1:
        raise ValueError('Require 0 <= low < high <= 1')
    if type(valid) is not bool:
        raise ValueError('valid must be a boolean')
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('Score must be null or a finite number in [0,1]')
    reason = 'UNCERTAIN_SCORE'
    decision = 'clarify'
    if language not in ('ro', 'uk'):
        reason = 'UNSUPPORTED_LANGUAGE'
    elif not valid or score is None:
        reason = 'INVALID_MODEL_OUTPUT'
    elif score >= high:
        decision, reason = 'review', 'CONCERN_PROPOSAL'
    elif score < low:
        decision, reason = 'no_signal', 'NO_SIGNAL_DETECTED'
    return dict(decision=decision, reason=reason, permits_external_action=False,
                establishes_safety=False)


def evaluate(expected, predictions):
    """Require complete identity matching; report strict and selective metrics."""
    truth = {}
    for row in expected:
        key = row['case_id']
        if not isinstance(key, str) or not key or key in truth:
            raise ValueError('Invalid or duplicate expected case ID')
        if type(row['gold']) is not int or row['gold'] not in (0, 1):
            raise ValueError('Gold must be integer 0 or 1')
        if row['language'] not in ('ro', 'uk'):
            raise ValueError('Expected language must be ro or uk')
        truth[key] = row
    if not truth:
        raise ValueError('No evaluation cases')
    predicted = {}
    for row in predictions:
        key = row['case_id']
        if key not in truth or key in predicted:
            raise ValueError('Unknown or duplicate prediction ID')
        if 'predicted' not in row:
            raise ValueError('Missing predicted field; use explicit null for abstention')
        value = row['predicted']
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError('Prediction must be integer 0, 1 or null')
        if 'language' in row and row['language'] != truth[key]['language']:
            raise ValueError('Prediction language mismatch')
        predicted[key] = value
    if set(predicted) != set(truth):
        raise ValueError('Missing predictions; do not silently exclude cases')
    groups = {}
    invariant = defaultdict(list)
    families = defaultdict(list)
    for key, row in truth.items():
        value, gold = predicted[key], row['gold']
        group = row['language']+'_'+row.get('kind', 'all')
        m = groups.setdefault(group, dict(total=0, positive=0, negative=0, decided=0,
            abstained=0, correct_decided=0, missed_or_abstained_concerns=0,
            false_review_or_abstained_benign=0, direct_misses=0, direct_false_reviews=0))
        m['total'] += 1; m['positive' if gold else 'negative'] += 1
        m['decided'] += value is not None; m['abstained'] += value is None
        m['correct_decided'] += value is not None and value == gold
        m['missed_or_abstained_concerns'] += gold == 1 and value != 1
        m['false_review_or_abstained_benign'] += gold == 0 and value != 0
        m['direct_misses'] += gold == 1 and value == 0
        m['direct_false_reviews'] += gold == 0 and value == 1
        if row.get('invariant_group'):
            invariant[(row['language'], row['invariant_group'])].append((gold, value))
        if row.get('pair_group'):
            families[(row['language'], row['pair_group'])].append((gold, value))
    for m in groups.values():
        m['coverage'] = m['decided']/m['total']
        m['decided_accuracy'] = m['correct_decided']/m['decided'] if m['decided'] else None
        m['strict_accuracy'] = m['correct_decided']/m['total']
        m['concern_recall_including_abstentions'] = 1-m['missed_or_abstained_concerns']/m['positive'] if m['positive'] else None
        m['benign_review_or_abstention_rate'] = m['false_review_or_abstained_benign']/m['negative'] if m['negative'] else None
    for values in invariant.values():
        if len({gold for gold, _ in values}) != 1:
            raise ValueError('Invariant group contains contradictory gold labels')
    inv = [v for v in invariant.values() if len(v) > 1]
    pairs = [v for v in families.values() if len({gold for gold, _ in v}) == 2]
    return dict(groups=groups, invariance=dict(groups=len(inv), inconsistent=sum(len({p for _,p in v})>1 for v in inv),
        fully_correct=sum(all(g==p for g,p in v) for v in inv)),
        paired_scenarios=dict(groups=len(pairs), fully_correct=sum(all(g==p for g,p in v) for v in pairs)),
        caveat='These metrics describe the supplied labeled inputs, not real-world safety.')


def target_failures(report):
    """Observed-count prerequisites only; no claim of fresh/stable readiness."""
    failures = []
    for name in ('ro_formal', 'ro_conversational', 'uk_formal', 'uk_conversational'):
        m = report.get('groups', {}).get(name)
        if not m or not m['positive'] or not m['negative']:
            failures.append(name+': both classes must be represented')
            continue
        if m['missed_or_abstained_concerns']*100 > m['positive']:
            failures.append(name+': concern recall below 99% including abstentions')
        if m['false_review_or_abstained_benign']*100 > 5*m['negative']:
            failures.append(name+': benign review/abstention rate above 5%')
    return failures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--require-targets',action='store_true',help='Exit nonzero unless every language/kind meets observed 99%% recall and 5%% false-review targets, counting abstentions')
    a=p.parse_args()
    result=evaluate([json.loads(l) for l in a.expected.read_text(encoding='utf-8').splitlines()],
                    [json.loads(l) for l in a.predictions.read_text(encoding='utf-8').splitlines()])
    result['target_failures']=target_failures(result)
    result['observed_targets_pass']=not result['target_failures']
    result['stable_release_ready']=False
    result['additional_checks']=['independent fresh evaluation', 'integrated runtime and serious-concern regressions']
    raw=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(raw, encoding='utf-8')
    print(raw,end='')
    if a.require_targets and result['target_failures']:
        raise SystemExit(1)

if __name__=='__main__':main()
