"""Exact, bounded decision clarification over declared interpretation worlds.

Certificates are reproducible computations, not signatures or permission grants.
The host authenticates reviewers and remains responsible for hypothesis coverage.
"""
import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

from .evidence import ONTOLOGY, _complete_action, analyze_evidence

FORMAT = 'roguard-assurance-v9-alpha1'
FIELDS = tuple(ONTOLOGY)
LABELS = {
    'ro': {'subject': 'Despre cine este vorba?', 'time': 'Când se petrece situația?',
           'assertion': 'Afirmația este confirmată, negată, incertă sau retrasă?',
           'source': 'Este o relatare directă, indirectă sau fictivă?',
           'relation': 'Mesajul este nou sau retrage o afirmație anterioară?',
           'concern': 'Există un semnal de îngrijorare în acest mesaj?'},
    'uk': {'subject': 'Про кого йдеться?', 'time': 'Коли відбувається ситуація?',
           'assertion': 'Твердження підтверджене, заперечене, непевне чи відкликане?',
           'source': 'Це пряма розповідь, переказ чи вигадка?',
           'relation': 'Це нове повідомлення чи відкликання попереднього твердження?',
           'concern': 'Чи є в цьому повідомленні сигнал занепокоєння?'},
}


def digest(value):
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('ascii')).hexdigest()


@lru_cache(maxsize=1)
def policy_digest():
    table = []
    for values in itertools.product(*(ONTOLOGY[field] for field in FIELDS)):
        world = dict(zip(FIELDS, values))
        table.append([world, _complete_action(world)])
    return digest(table)


def normalize(problem):
    required = {'language', 'hypotheses', 'context_digest'}
    optional = {'question_costs', 'review_floor', 'model_digests'}
    if not isinstance(problem, dict) or not required <= set(problem) or set(problem) - required - optional:
        raise ValueError('Expected language, hypotheses and context_digest')
    if problem['language'] not in LABELS:
        raise ValueError('Unsupported language')
    context = problem['context_digest']
    if not isinstance(context, str) or len(context) != 64 or any(c not in '0123456789abcdef' for c in context):
        raise ValueError('context_digest must be a lowercase SHA-256 digest')
    worlds = problem['hypotheses']
    if not isinstance(worlds, list) or not 1 <= len(worlds) <= 512:
        raise ValueError('Expected 1 to 512 joint hypotheses')
    unique = set()
    for world in worlds:
        if not isinstance(world, dict) or set(world) != set(FIELDS):
            raise ValueError('Every hypothesis must contain all six attributes')
        if any(not isinstance(world[field], str) or world[field] not in ONTOLOGY[field] for field in FIELDS):
            raise ValueError('Invalid attribute value')
        unique.add(tuple(world[field] for field in FIELDS))
    costs = problem.get('question_costs', dict.fromkeys(FIELDS, 1))
    if not isinstance(costs, dict) or set(costs) - set(FIELDS):
        raise ValueError('Unknown question')
    if any(type(value) is not int or not 1 <= value <= 1000 for value in costs.values()):
        raise ValueError('Question costs must be integers from 1 to 1000')
    floor = problem.get('review_floor', False)
    if type(floor) is not bool:
        raise ValueError('review_floor must be boolean')
    models = problem.get('model_digests', [])
    if not isinstance(models, list) or len(models) > 8 or any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in models):
        raise ValueError('Expected up to eight model SHA-256 digests')
    return dict(language=problem['language'], context_digest=context,
                hypotheses=[dict(zip(FIELDS, values)) for values in sorted(unique)],
                question_costs=dict(sorted(costs.items())), review_floor=floor,
                model_digests=sorted(set(models)))


def action(world, floor=False):
    result = _complete_action(world)
    return 'REVIEW_AND_CONTINUE' if floor and result == 'CONTINUE' else result


def assure(problem):
    normalized = normalize(problem)
    worlds = normalized['hypotheses']
    actions = [action(world, normalized['review_floor']) for world in worlds]
    costs = normalized['question_costs']

    @lru_cache(maxsize=None)
    def solve(indices):
        possible = sorted({actions[i] for i in indices})
        if len(possible) == 1:
            return 0, {'action': possible[0], 'hypotheses': len(indices)}
        candidates = []
        for field, cost in costs.items():
            values = sorted({worlds[i][field] for i in indices})
            if len(values) < 2:
                continue
            branches = {value: solve(tuple(i for i in indices if worlds[i][field] == value)) for value in values}
            if any(result[0] is None for result in branches.values()):
                continue
            worst = cost + max(result[0] for result in branches.values())
            candidates.append((worst, field, {value: result[1] for value, result in branches.items()}))
        if not candidates:
            return None, None
        worst, field, branches = min(candidates, key=lambda candidate: candidate[:2])
        return worst, {'question': field, 'label': LABELS[normalized['language']][field],
                       'cost': costs[field], 'branches': branches}

    worst, tree = solve(tuple(range(len(worlds))))
    result = dict(format=FORMAT, input=normalized, input_digest=digest(normalized),
                  policy_digest=policy_digest(), possible_actions=sorted(set(actions)),
                  optimal_worst_case_cost=worst, tree=tree, states_evaluated=solve.cache_info().currsize,
                  status='ACTION_DETERMINED' if len(set(actions)) == 1 else 'CLARIFICATION_REQUIRED' if tree else 'UNRESOLVED',
                  coverage_verified=False, reviewer_authenticated=False, permits_external_action=False,
                  limitation='Exact only for supplied hypotheses, question costs and current policy; no safety or accuracy guarantee.')
    result['certificate_digest'] = digest(result)
    return result


def verify(certificate, *, expected_input_digest=None):
    """Recompute, including tree, optimal costs, policy and all metadata."""
    if not isinstance(certificate, dict) or 'input' not in certificate:
        raise ValueError('Invalid assurance certificate')
    expected = assure(certificate['input'])
    # Canonical bytes distinguish booleans from numbers, unlike Python equality.
    if digest(certificate) != digest(expected):
        raise ValueError('Certificate does not match recomputed decision')
    if expected_input_digest is not None and expected['input_digest'] != expected_input_digest:
        raise ValueError('Certificate belongs to another input')
    return True


def answer(certificate, answers, *, expected_certificate_digest):
    """Host-only clarification; callers must authenticate the reviewer separately."""
    verify(certificate)
    if certificate['certificate_digest'] != expected_certificate_digest:
        raise ValueError('Stale clarification binding')
    if not isinstance(answers, dict) or not answers or set(answers) - set(FIELDS):
        raise ValueError('Expected nonempty attribute answers')
    if any(not isinstance(value, str) or value not in ONTOLOGY[field] for field, value in answers.items()):
        raise ValueError('Invalid reviewer answer')
    problem = deepcopy(certificate['input'])
    if set(answers) - set(problem['question_costs']):
        raise ValueError('Question is not available for clarification')
    worlds = [world for world in problem['hypotheses'] if all(world[field] == value for field, value in answers.items())]
    if not worlds:
        return {'status': 'OUTSIDE_HYPOTHESES', 'parent_digest': expected_certificate_digest,
                'requires_reinterpretation': True, 'permits_external_action': False}
    problem['hypotheses'] = worlds
    return {'status': 'ANSWER_APPLIED', 'parent_digest': expected_certificate_digest,
            'answers': deepcopy(answers), 'certificate': assure(problem),
            'reviewer_authenticated': False, 'permits_external_action': False}


def combine(certificates):
    """Retain the union of complete worlds; never invent cross-model hybrids."""
    if not isinstance(certificates, list) or not 1 <= len(certificates) <= 8:
        raise ValueError('Expected one to eight certificates')
    for certificate in certificates:
        verify(certificate)
    problem = deepcopy(certificates[0]['input'])
    for certificate in certificates[1:]:
        other = certificate['input']
        if any(other[key] != problem[key] for key in ('language','context_digest','question_costs')):
            raise ValueError('Cannot combine different contexts, languages or costs')
        problem['hypotheses'].extend(other['hypotheses'])
        problem['model_digests'].extend(other['model_digests'])
        problem['review_floor'] |= other['review_floor']
    problem['hypotheses'] = [dict(zip(FIELDS, values)) for values in sorted({tuple(w[f] for f in FIELDS) for w in problem['hypotheses']})]
    problem['model_digests'] = sorted(set(problem['model_digests']))
    return assure(problem)


def assure_conversation(payload, model=None):
    """Use actual evidence hypotheses, retaining their existing Cartesian semantics."""
    report = analyze_evidence(payload, model)
    context = digest(payload)
    entries = []
    for evidence in report['evidence']:
        hypotheses = evidence['action_envelope']['effective_hypotheses']
        worlds = [dict(zip(FIELDS, values)) for values in itertools.product(*(hypotheses[field] for field in FIELDS))]
        floor = bool(evidence.get('review_floor') and evidence['review_floor']['review_recommended'])
        certificate = assure(dict(language=payload['language'], context_digest=digest({'context': context, 'reference': evidence['reference']}),
                                 hypotheses=worlds, review_floor=floor, model_digests=[report['model_identity']]))
        entries.append(dict(id=evidence['id'], reference=evidence['reference'], certificate=certificate))
    return dict(format=FORMAT, language=payload['language'], model_identity=report['model_identity'],
                context_digest=context, entries=entries, text_in_report=False, permits_external_action=False,
                hypothesis_source='existing evidence prediction sets; coverage unverified')


def read_json(path):
    with Path(path).open('rb') as source:
        raw = source.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('Assurance input exceeds 2 MB')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('Nonfinite JSON value')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=reject_constant)


def main():
    parser = argparse.ArgumentParser(description='Local decision assurance; no authorization or external actions')
    parser.add_argument('input', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--conversation', action='store_true', help='Interpret a conversation using the existing experimental model')
    mode.add_argument('--verify', action='store_true', help='Recompute an existing certificate')
    mode.add_argument('--answers', type=Path, help='Apply host-reviewed attribute answers to a certificate')
    mode.add_argument('--combine', action='store_true', help='Input is a list of certificates for the same context')
    parser.add_argument('--binding', help='Exact parent certificate digest, required with --answers')
    args = parser.parse_args()
    if bool(args.answers) != bool(args.binding):
        parser.error('--answers and --binding must be supplied together')
    try:
        payload = read_json(args.input)
        if args.answers:
            result = answer(payload, read_json(args.answers), expected_certificate_digest=args.binding)
        elif args.combine:
            result = combine(payload)
        else:
            result = {'verified_computation': verify(payload), 'permits_external_action': False} if args.verify else assure_conversation(payload) if args.conversation else assure(payload)
        print(json.dumps(result, ensure_ascii=True, indent=2))
    except (ValueError, KeyError, TypeError, OSError, RecursionError) as error:
        parser.exit(2, 'Invalid assurance input: '+str(error)+'\n')


if __name__ == '__main__':
    main()
