"""Packaged interaction laboratory API and CLI; browser routes are offline."""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path
from .cli import _strict_json_loads
from .flow import integer
from .interaction_lab import ReferenceTarget, run_scenario, minimize_failure, recorded_factory
from .lab_scenarios import FAMILIES, make_scenario, generate_suite
from .lab_metrics import summarize_runs, tradeoff_frontier
from .consistency import evaluate_consistency, conversation_consistency


def policy_pairs(language='ro', seed=17):
    base = make_scenario(language, 'ordinary_help', seed)
    pairs = []
    for name, key, value in [('recipient', 'recipient', 'guardian'), ('purpose', 'purpose', 'progress_report'), ('operation', 'operation', 'memory')]:
        changed = copy.deepcopy(base); changed['events'][-1][key] = value; changed['events'][-1]['expected'] = 'BLOCK'
        pairs.append({'pair_id': name, 'relation': 'transition', 'before': base, 'after': changed,
                      'expected': ['ALLOW', 'BLOCK']})
    revoked = copy.deepcopy(base)
    revoked['events'].insert(-1, {'id': 'revoke_event', 'kind': 'permission', 'source': 'note', 'purpose': 'tutoring',
                                 'recipient': 'student', 'operation': 'use', 'action': 'revoke'})
    revoked['events'][-1]['expected'] = 'BLOCK'
    pairs.append({'pair_id': 'revocation', 'relation': 'transition', 'before': base, 'after': revoked, 'expected': ['ALLOW', 'BLOCK']})
    upper = copy.deepcopy(base)
    for event in upper['events']:
        for key in ('text', 'value'):
            if key in event: event[key] = event[key].upper()
    other = make_scenario('uk' if language == 'ro' else 'ro', 'ordinary_help', seed)
    for name, after in [('case', upper), ('language', other)]:
        pairs.append({'pair_id': name, 'relation': 'invariant', 'before': base, 'after': after, 'expected': ['ALLOW', 'ALLOW']})
    return pairs


def run_suite(scenarios, target_factory=None, *, enforce=True):
    if not isinstance(scenarios, list) or not 1 <= len(scenarios) <= 2400:
        raise ValueError('Provide 1 to 2400 scenarios')
    # Validate all scenario syntax before any user-supplied adapter is invoked.
    from .interaction_lab import validate_scenario
    for scenario in scenarios: validate_scenario(scenario)
    if len({s['case_id'] for s in scenarios}) != len(scenarios):
        raise ValueError('Suite case IDs must be unique')
    runs = [run_scenario(s, target_factory, enforce=enforce) for s in scenarios]
    return {'schema_version': '5.0', 'summary': summarize_runs(runs),
            'by_language': {language: summarize_runs([r for r in runs if r['language'] == language])
                            for language in ('ro', 'uk') if any(r['language'] == language for r in runs)},
            'failures': [{'case_id': r['case_id'], **r['first_failure']} for r in runs if r['first_failure']],
            'external_action_taken': False, 'text_in_report': False}


def workbench(payload):
    if not isinstance(payload, dict) or set(payload) - {'language', 'family', 'seed', 'target', 'enforce', 'reduce', 'scenario', 'replies'}:
        raise ValueError('Invalid laboratory request')
    language = payload.get('language', 'ro'); family = payload.get('family', 'revocation')
    seed = payload.get('seed', 17); target = payload.get('target', 'guarded')
    enforce = payload.get('enforce', True); reduce = payload.get('reduce', False)
    integer(seed, 'seed', 2**32-1)
    if language not in ('ro', 'uk') or family not in FAMILIES or target not in ('guarded', 'permissive', 'overblocking', 'stale'):
        raise ValueError('Unknown laboratory configuration')
    if type(enforce) is not bool or type(reduce) is not bool:
        raise ValueError('Enforcement and reduction must be booleans')
    scenario = payload['scenario'] if 'scenario' in payload else make_scenario(language, family, seed)
    if not isinstance(scenario, dict):
        raise ValueError('Scenario must be an object')
    if scenario.get('language') != language:
        raise ValueError('Scenario language must match the workspace')
    factory = recorded_factory(payload['replies']) if 'replies' in payload else lambda: ReferenceTarget(target)
    run = run_scenario(scenario, factory, enforce=enforce)
    result = {'schema_version': '5.0', 'run': run, 'metrics': summarize_runs([run]),
              'target': 'recorded' if 'replies' in payload else target,
              'reproducer': minimize_failure(scenario, factory, enforce=enforce) if reduce else None,
              'external_action_taken': False}
    # Recorded replies correspond to the user's one scenario, not separate paired cases.
    if 'replies' not in payload and 'scenario' not in payload:
        pairs = policy_pairs(language, seed)
        result['target_consistency'] = evaluate_consistency(pairs, lambda s: run_scenario(s, factory, enforce=False)['requests'][-1]['proposal'])
        result['enforced_consistency'] = evaluate_consistency(pairs, lambda s: run_scenario(s, factory, enforce=True)['requests'][-1]['outcome'])
    else:
        result['target_consistency'] = result['enforced_consistency'] = None
    result['text_in_report'] = bool(result['reproducer'] and result['reproducer'].get('text_in_report'))
    return result


def compare_reference_targets(language='ro', seed=17):
    scenarios = [s for s in generate_suite(seed) if s['language'] == language]
    if not scenarios:
        raise ValueError('Unknown language')
    results = {}; points = []
    for mode, enforce in [('permissive', False), ('permissive', True), ('overblocking', True), ('guarded', True), ('stale', True)]:
        name = mode + ('_enforced' if enforce else '_unrestricted')
        report = run_suite(scenarios, lambda: ReferenceTarget(mode), enforce=enforce)
        results[name] = report['summary']
        points.append({'name': name, 'unauthorized_rate': report['summary']['unauthorized_deliveries']['rate'],
                       'utility_rate': report['summary']['useful_task_preservation']['rate']})
    return {'results': results, 'tradeoff': tradeoff_frontier(points),
            'controlled_reference_targets': True, 'language_model_accuracy_established': False,
            'text_in_report': False}


def _read(path):
    with Path(path).open(encoding='utf-8') as stream: raw = stream.read(4_000_001)
    if len(raw) > 4_000_000: raise ValueError('Input too large')
    return _strict_json_loads(raw)


def main():
    parser = argparse.ArgumentParser(description='Local fictional interaction lab; never dispatches external tools')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('demo', 'run', 'reduce', 'suite'):
        command = sub.add_parser(name)
        command.add_argument('--target', choices=['guarded', 'permissive', 'overblocking', 'stale'], default='guarded')
        command.add_argument('--no-enforce', action='store_true')
        if name != 'reduce': command.add_argument('--fail-on-failure', action='store_true')
        if name in ('run', 'reduce'):
            command.add_argument('input'); command.add_argument('--recorded')
        else:
            command.add_argument('--seed', type=int, default=17)
        if name == 'demo':
            command.add_argument('--language', choices=['ro', 'uk'], default='ro')
            command.add_argument('--family', choices=FAMILIES, default='revocation')
        if name == 'suite': command.add_argument('--repetitions', type=int, default=1)
        if name == 'reduce': command.add_argument('--max-probes', type=int, default=96)
    compare = sub.add_parser('compare'); compare.add_argument('--language', choices=['ro', 'uk'], default='ro'); compare.add_argument('--seed', type=int, default=17)
    consistent = sub.add_parser('consistency'); consistent.add_argument('input')
    consistent.add_argument('--fail-on-failure', action='store_true')
    consistent.add_argument('--backend', choices=['fast', 'semantic', 'hybrid'], default='fast')
    args = parser.parse_args()
    try:
        if args.command == 'compare':
            result = compare_reference_targets(args.language, args.seed)
        elif args.command == 'consistency':
            from .conversation import ConversationScorer
            result = conversation_consistency(_read(args.input), ConversationScorer(args.backend))
        else:
            factory = recorded_factory(_read(args.recorded)) if getattr(args, 'recorded', None) else lambda: ReferenceTarget(args.target)
            if args.command == 'suite':
                result = run_suite(generate_suite(args.seed, args.repetitions), factory, enforce=not args.no_enforce)
            else:
                scenario = make_scenario(args.language, args.family, args.seed) if args.command == 'demo' else _read(args.input)
                if args.command == 'reduce': result = minimize_failure(scenario, factory, enforce=not args.no_enforce, max_probes=args.max_probes)
                else: result = run_scenario(scenario, factory, enforce=not args.no_enforce)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if getattr(args, 'fail_on_failure', False) and (result.get('first_failure') or result.get('failures') or result.get('failed') or result.get('unresolved')):
            raise SystemExit(1)
    except (ValueError, TypeError, KeyError, OSError, ImportError):
        parser.error('Invalid laboratory input or unavailable optional model; see the interaction laboratory guide')

if __name__ == '__main__': main()
