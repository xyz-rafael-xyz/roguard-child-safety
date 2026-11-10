"""Strict trajectory evaluation; failed/abstained cases remain in denominators."""
from __future__ import annotations

from collections import defaultdict
import math
import random
import statistics

from .assurance import digest
from .trajectory_data import validate_case, prefix, input_identity


def checkpoint_id(case, point):
    return case['case_id']+':'+str(point['focus'])


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(q*len(values))-1)]


def evaluate(cases, predictions, *, bootstrap=1000, seed=1010):
    if type(bootstrap) is not int or not 0 <= bootstrap <= 10000:
        raise ValueError('Invalid bootstrap repetition count')
    expected, case_map = {}, {}
    for case in cases:
        validate_case(case)
        if case['case_id'] in case_map:
            raise ValueError('Duplicate case')
        case_map[case['case_id']] = case
        for point in case['checkpoints']:
            expected[checkpoint_id(case, point)] = case, point
    if not expected:
        raise ValueError('No expected predictions')
    indexed = {}
    for row in predictions:
        key = row['checkpoint_id']
        if key not in expected or key in indexed:
            raise ValueError('Unexpected or duplicate prediction')
        case, point = expected[key]
        if row['input_digest'] != input_identity(prefix(case, point['focus'])) or row['language'] != case['language'] or row['focus'] != point['focus']:
            raise ValueError('Prediction is not bound to the evaluated prefix')
        value = row['predicted']
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError('Invalid prediction; use null for abstention')
        for name in ('latency_seconds',):
            v = row.get(name)
            if v is not None and (type(v) not in (float, int) or not math.isfinite(v) or v < 0):
                raise ValueError('Invalid timing')
        spans = row.get('evidence', [])
        keys = set()
        for span in spans:
            t, a, b = span['turn'], span['start'], span['end']
            if any(type(v) is not int for v in (t, a, b)) or not 0 <= t <= point['focus'] or not 0 <= a < b <= len(case['turns'][t]['text']) or (t,a,b) in keys:
                raise ValueError('Invalid, repeated or future evidence')
            keys.add((t,a,b))
        indexed[key] = row
    if set(indexed) != set(expected):
        raise ValueError('Missing predictions; record explicit abstentions')
    groups = defaultdict(list)
    for key, (case, point) in expected.items():
        groups[case['language']].append((case, point, indexed[key]))
    reports = {}
    for lang, rows in groups.items():
        positives = [(c,p,r) for c,p,r in rows if p['review'] == 1]
        negatives = [(c,p,r) for c,p,r in rows if p['review'] == 0]
        decided = sum(r['predicted'] is not None for _,_,r in rows)
        correct = sum(r['predicted'] == p['review'] for _,p,r in rows)
        family_scores = defaultdict(list)
        covered, sizes, gold_spans, retained = [], [], 0, 0
        reference_candidates = 0
        supported_candidates = 0
        budgets = []
        for c,p,r in rows:
            family_scores[c['family']].append(int(r['predicted'] == p['review']))
            if 'interpretation_set' in r and p.get('world') is not None:
                worlds = r['interpretation_set']['hypotheses']
                from .trajectory_sets import WORLD_IDS
                if not worlds or len(worlds)>432 or len({digest(w) for w in worlds})!=len(worlds) or any(digest(w) not in WORLD_IDS for w in worlds):
                    raise ValueError('Malformed interpretation prediction set')
                covered.append(p['world'] in worlds)
                sizes.append(len(worlds))
            for span in p.get('evidence', []):
                gold_spans += 1
                retained += any(s['turn'] == span['turn'] and s['start'] <= span['start'] and s['end'] >= span['end'] for s in r.get('evidence', []))
            # Candidate precision measures overlap with annotation, not truth.
            if 'evidence' in p and 'evidence' in r:
                for s in r['evidence']:
                    reference_candidates += 1
                    supported_candidates += any(g['turn'] == s['turn'] and max(g['start'],s['start']) < min(g['end'],s['end']) for g in p['evidence'])
            if r.get('assurance', {}).get('optimal_worst_case_cost') is not None:
                budgets.append(r['assurance']['optimal_worst_case_cost'])
        by_case = defaultdict(list)
        for c,p,r in rows:
            by_case[c['case_id']].append((p,r))
        benign, false_alerts, burden, eligible, detected, delays, premature = 0,0,0,0,0,[],0
        for items in by_case.values():
            items.sort(key=lambda v: v[0]['focus'])
            first = next((p['focus'] for p,r in items if p['review']), None)
            if first is None:
                benign += 1
                false_alerts += any(r['predicted'] == 1 for p,r in items)
                burden += any(r['predicted'] != 0 for p,r in items)
            else:
                eligible += 1
                premature += any(r['predicted'] == 1 and p['focus'] < first for p,r in items)
                hit = next((p['focus'] for p,r in items if p['focus'] >= first and p['review'] == 1 and r['predicted'] == 1), None)
                if hit is not None:
                    detected += 1
                    delays.append(hit-first)
        # Bootstrap whole scenario families; this is descriptive, not proof of
        # independence. All correlated prefixes and languages retain family IDs.
        rng = random.Random(seed)
        family_values = list(family_scores.values())
        boot = []
        if len(family_values) > 1:
            for _ in range(bootstrap):
                sample = [v for group in rng.choices(family_values, k=len(family_values)) for v in group]
                boot.append(sum(sample)/len(sample))
        reports[lang] = {
            'checkpoints': len(rows), 'conversations': len(by_case), 'families': len(family_values),
            'strict_accuracy': correct/len(rows), 'coverage': decided/len(rows),
            'abstentions': len(rows)-decided, 'positive_checkpoints': len(positives), 'negative_checkpoints': len(negatives),
            'recall_including_abstentions': sum(r['predicted']==1 for _,_,r in positives)/len(positives) if positives else None,
            'benign_checkpoint_burden': sum(r['predicted']!=0 for _,_,r in negatives)/len(negatives) if negatives else None,
            'benign_conversations': benign,
            'false_alerts_per_1000_benign_conversations': 1000*false_alerts/benign if benign else None,
            'review_or_abstention_per_1000_benign_conversations': 1000*burden/benign if benign else None,
            'eligible_conversations': eligible, 'detected_conversations': detected, 'missed_conversations': eligible-detected,
            'premature_alert_conversations': premature,
            'mean_detection_delay_turns_among_detected': statistics.mean(delays) if delays else None,
            'delay_scope': 'annotated checkpoints only; misses reported separately',
            'joint_set_coverage': sum(covered)/len(covered) if covered else None,
            'outside_ontology_cases':sum(bool(p.get('outside_ontology')) for _,p,_ in rows),
            'outside_ontology_abstentions':sum(bool(p.get('outside_ontology')) and r['predicted'] is None for _,p,r in rows),
            'mean_set_size': statistics.mean(sizes) if sizes else None,
            'annotated_evidence_retention': retained/gold_spans if gold_spans else None,
            'candidate_evidence_overlap_precision': supported_candidates/reference_candidates if reference_candidates else None,
            'mean_declared_question_cost': statistics.mean(budgets) if budgets else None,
            'p95_latency_seconds': percentile([r['latency_seconds'] for _,_,r in rows if r.get('latency_seconds') is not None], .95),
            'family_bootstrap_accuracy_95_interval': [percentile(boot,.025),percentile(boot,.975)] if boot else None,
        }
    return {'format': 'roguard-trajectory-evaluation-v10-research1', 'groups': reports,
            'cases_digest': digest(cases), 'predictions_digest': digest(predictions),
            'synthetic_only': all(c['synthetic'] for c in cases), 'independently_validated': False,
            'stable_release_ready': False, 'interpretation': 'Descriptive experiment; no measured child benefit.'}


def paired_compare(cases, baseline, candidate, *, bootstrap=1000, seed=1010):
    evaluate(cases, baseline, bootstrap=0)
    evaluate(cases, candidate, bootstrap=0)
    left = {r['checkpoint_id']: r for r in baseline}
    right = {r['checkpoint_id']: r for r in candidate}
    families = defaultdict(list)
    wins = losses = 0
    for case in cases:
        for point in case['checkpoints']:
            key = checkpoint_id(case, point)
            a = left[key]['predicted'] == point['review']
            b = right[key]['predicted'] == point['review']
            families[case['family']].append(int(b)-int(a))
            wins += b and not a
            losses += a and not b
    values = list(families.values())
    rng = random.Random(seed)
    samples = []
    for _ in range(bootstrap):
        sample = [x for group in rng.choices(values, k=len(values)) for x in group]
        samples.append(sum(sample)/len(sample))
    flat = [x for group in values for x in group]
    return {'candidate_only_correct': wins, 'baseline_only_correct': losses,
            'strict_accuracy_delta': sum(flat)/len(flat),
            'paired_family_bootstrap_delta_95_interval': [percentile(samples,.025),percentile(samples,.975)],
            'comparison_digest': digest([baseline,candidate]), 'real_world_superiority_established': False}
