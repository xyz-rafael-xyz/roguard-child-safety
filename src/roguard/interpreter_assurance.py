"""Bind structured interpreter proposals to the local clarification planner.

Hashes bind data; they do not authenticate a model service or prove coverage.
Hosts must obtain reports from their own trusted inference boundary.
"""
import itertools
import json
import math

from .assurance import assure, digest
from .evidence import ONTOLOGY, disposition
from .interpreter import CONTRACT_DIGEST
from .native_review import prepare_input


def plan_review(payload, reports, *, expected_model_identities, focus=None, question_costs=None):
    """Preserve joint model interpretations; abstention keeps all 432 worlds.

    ``expected_model_identities`` comes from host configuration, not from an
    untrusted request. No probabilities are interpreted as calibrated coverage.
    """
    _, focus, context_digest = prepare_input(payload, focus)
    if not isinstance(reports, list) or not 1 <= len(reports) <= 8:
        raise ValueError('Expected one to eight interpreter reports')
    if (not isinstance(expected_model_identities, (list, tuple)) or not 1 <= len(expected_model_identities) <= 8
            or any(not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                   for value in expected_model_identities)
            or len(set(expected_model_identities)) != len(expected_model_identities)):
        raise ValueError('Expected distinct configured model SHA-256 identities')
    models, worlds, report_digests, abstained = [], [], [], False
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError('Expected interpreter report object')
        # Bound and canonicalize before hashing; reject unsupported Python values.
        encoded = json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
        if len(encoded) > 32768:
            raise ValueError('Interpreter report exceeds 32768 characters')
        if (report.get('format') != 'roguard-interpreter-v1' or report.get('language') != payload['language']
                or type(report.get('focus')) is not int or report['focus'] != focus
                or report.get('input_digest') != context_digest or report.get('contract_digest') != CONTRACT_DIGEST):
            raise ValueError('Interpreter report belongs to another input or contract')
        identity = report.get('model_identity')
        if identity not in expected_model_identities or identity in models:
            raise ValueError('Unexpected or repeated model identity')
        for flag in ('probabilities_are_calibrated', 'text_in_report', 'establishes_safety',
                     'permits_external_action', 'permits_notification'):
            if report.get(flag) is not False:
                raise ValueError('Unsupported interpreter claim: '+flag)
        valid = report.get('valid_label_tokens')
        if type(valid) is not bool:
            raise ValueError('Invalid label validity flag')
        values = report.get('interpretation')
        probabilities = report.get('conditional_field_probabilities')
        if values is not None:
            if not isinstance(values, dict) or set(values) != set(ONTOLOGY):
                raise ValueError('Incomplete interpretation')
            if not isinstance(probabilities, dict) or set(probabilities) != set(ONTOLOGY):
                raise ValueError('Incomplete conditional probabilities')
            for field, labels in ONTOLOGY.items():
                p = probabilities[field]
                if (values[field] not in labels or not isinstance(p, list) or len(p) != len(labels)
                        or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                        or abs(sum(p)-1) > 1e-4 or labels[max(range(len(p)), key=p.__getitem__)] != values[field]):
                    raise ValueError('Interpretation differs from conditional probabilities')
        elif probabilities is not None:
            raise ValueError('Probabilities without interpretation')
        if valid:
            if values is None:
                raise ValueError('Valid report requires an interpretation')
            prediction = int(disposition({k: {'label': v} for k, v in values.items()})[0] != 'CONTINUE')
            if type(report.get('predicted')) is not int or report['predicted'] != prediction:
                raise ValueError('Review proposal differs from interpretation')
            worlds.append(dict(values))
        else:
            if report.get('predicted') is not None:
                raise ValueError('An invalid interpretation must abstain')
            abstained = True
        models.append(identity); report_digests.append(digest(report))
    if set(models) != set(expected_model_identities):
        raise ValueError('A configured model report is missing')
    if abstained:
        worlds = [dict(zip(ONTOLOGY, values)) for values in itertools.product(*ONTOLOGY.values())]
    problem = dict(language=payload['language'], context_digest=context_digest,
                   hypotheses=worlds, model_digests=models)
    if question_costs is not None:
        problem['question_costs'] = question_costs
    return dict(format='roguard-interpreter-assurance-v1', focus=focus, context_digest=context_digest,
                contract_digest=CONTRACT_DIGEST, source_report_digests=sorted(report_digests),
                source_models=sorted(models), includes_abstention=abstained, certificate=assure(problem),
                reports_authenticated=False, hypothesis_coverage_verified=False,
                probabilities_used_for_coverage=False, text_in_report=False,
                permits_external_action=False, permits_notification=False)


def verify_review_plan(result, payload, reports, *, expected_model_identities, focus=None, question_costs=None):
    """Recompute both report binding and the contained clarification certificate."""
    expected = plan_review(payload, reports, expected_model_identities=expected_model_identities,
                           focus=focus, question_costs=question_costs)
    if digest(result) != digest(expected):
        raise ValueError('Review plan differs from its bound reports and configuration')
    return True


def main():
    import argparse
    from .assurance import read_json
    parser = argparse.ArgumentParser(description='Plan local clarification from input-bound interpreter reports.')
    parser.add_argument('input', help='JSON containing conversation, reports and optional focus/question_costs')
    parser.add_argument('--model-sha256', action='append', required=True, help='Expected model identity from trusted configuration; repeat for multiple models')
    args = parser.parse_args()
    try:
        data = read_json(args.input)
        if (not isinstance(data, dict) or not {'conversation', 'reports'} <= set(data)
                or set(data)-{'conversation', 'reports', 'focus', 'question_costs'}):
            raise ValueError('Expected conversation, reports and optional focus/question_costs')
        result = plan_review(data['conversation'], data['reports'], expected_model_identities=args.model_sha256,
                             focus=data.get('focus'), question_costs=data.get('question_costs'))
        print(json.dumps(result, ensure_ascii=True, indent=2))
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as error:
        parser.exit(2, 'Invalid interpreter review input: '+str(error)+'\n')


if __name__ == '__main__':
    main()
