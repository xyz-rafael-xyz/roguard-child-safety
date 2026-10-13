import copy
import json
import unittest
from roguard.interaction_lab import ReferenceTarget, run_scenario, minimize_failure, recorded_factory, validate_scenario
from roguard.lab_scenarios import make_scenario, generate_suite
from roguard.lab import workbench, run_suite, policy_pairs, compare_reference_targets
from roguard.lab_metrics import summarize_runs, tradeoff_frontier
from roguard.consistency import evaluate_consistency


class InteractionTests(unittest.TestCase):
    def test_generated_scenarios_match_public_schema(self):
        try:import jsonschema
        except ImportError:self.skipTest('Optional schema validator unavailable')
        from pathlib import Path
        schema=json.loads((Path(__file__).resolve().parents[1]/'schema/interaction-scenario.schema.json').read_text())
        for scenario in generate_suite():jsonschema.validate(scenario,schema)

    def test_all_declared_reference_paths_preserve_authorized_utility(self):
        report=run_suite(generate_suite())
        self.assertFalse(report['failures'])
        self.assertEqual(report['summary']['reference_policy_conflicts'],0)
        self.assertEqual(report['summary']['useful_task_preservation']['rate'],1)
        self.assertEqual(report['summary']['unauthorized_deliveries']['count'],0)

    def test_unsafe_proposals_stopped_without_calling_them_safe(self):
        scenario=make_scenario('uk','revocation')
        raw=run_scenario(scenario,lambda:ReferenceTarget('permissive'),enforce=False)
        guarded=run_scenario(scenario,lambda:ReferenceTarget('permissive'))
        self.assertEqual(raw['requests'][-1]['outcome'],'ALLOW')
        self.assertEqual(guarded['requests'][-1]['outcome'],'BLOCK')
        self.assertTrue(guarded['requests'][-1]['unauthorized_proposal'])
        self.assertEqual(raw['first_failure'],guarded['first_failure'])
        self.assertFalse(guarded['external_action_taken'])

    def test_no_proposal_can_modify_authority_or_execute_code(self):
        p=make_scenario('ro','ordinary_help')
        for reply in [{'decision':'ALLOW','selected':['summary'],'grant':True},
                      {'decision':'ALLOW','selected':['unregistered']},
                      {'decision':'BLOCK','selected':['summary']},None,{'decision':[],'selected':[]}]:
            r=run_scenario(p,recorded_factory([reply]))
            self.assertEqual(r['first_failure']['code'],'INVALID_TARGET_OUTPUT')
            self.assertEqual(r['requests'][0]['outcome'],'REVIEW')
        with self.assertRaises(ValueError):workbench({'target_url':'https://example.org'})

    def test_target_exceptions_are_unresolved_without_echoing_text(self):
        def factory():
            def fail(observation):raise RuntimeError('SECRET ERROR CONTENT')
            return fail
        r=run_scenario(make_scenario(),factory)
        self.assertEqual(r['first_failure']['code'],'TARGET_ERROR')
        self.assertNotIn('SECRET ERROR CONTENT',json.dumps(r))
        self.assertTrue(r['requests'][0]['unresolved'])

    def test_observations_cannot_mutate_ledger_or_scenario(self):
        scenario=make_scenario('ro','ordinary_help');original=copy.deepcopy(scenario)
        def factory():
            def target(observation):
                observation['request']['recipient']='guardian'
                observation['disclosure_proposal']['selected'].append('forged')
                return {'decision':'ALLOW','selected':['summary'],'answer':-10}
            return target
        report=run_scenario(scenario,factory)
        self.assertEqual(scenario,original)
        self.assertEqual(report['requests'][0]['outcome'],'REVIEW')

    def test_wrong_arithmetic_is_not_useful_completion(self):
        report=run_scenario(make_scenario('ro','ordinary_help'),recorded_factory([{'decision':'ALLOW','selected':['summary'],'answer':0}]))
        summary=summarize_runs([report])
        self.assertEqual(summary['checkable_task_success']['count'],0)
        self.assertEqual(summary['useful_task_preservation']['count'],0)
        self.assertEqual(summary['unauthorized_proposals']['count'],0)

    def test_reducer_replays_same_failure_and_removes_only_valid_dependencies(self):
        scenario=make_scenario('uk','revocation')
        factory=lambda:ReferenceTarget('stale')
        result=minimize_failure(scenario,factory)
        self.assertLess(result['events_after'],result['events_before'])
        self.assertTrue(result['one_deletion_minimal_among_valid_scenarios'])
        self.assertEqual(run_scenario(result['scenario'],factory)['first_failure'],result['failure'])
        for event in result['scenario']['events']:
            if event['id']==result['failure']['event_id']:continue
            trial=copy.deepcopy(result['scenario']);trial['events'].remove(event)
            try:r=run_scenario(trial,factory)
            except (ValueError,KeyError,TypeError):continue
            self.assertNotEqual(r['first_failure'],result['failure'])
        budget=minimize_failure(scenario,factory,max_probes=2)
        self.assertEqual(budget['probes'],2);self.assertFalse(budget['one_deletion_minimal_among_valid_scenarios'])

    def test_nondeterministic_target_does_not_get_minimality_claim(self):
        calls=[]
        def factory():calls.append(1);return ReferenceTarget('permissive' if len(calls)%2 else 'guarded')
        self.assertEqual(minimize_failure(make_scenario(),factory)['status'],'NONDETERMINISTIC_TARGET')

    def test_branch_paths_are_explicit_and_no_backward_edges(self):
        p=make_scenario('uk','branching')
        guarded=run_scenario(p);raw=run_scenario(p,lambda:ReferenceTarget('permissive'),enforce=False)
        self.assertNotEqual([r['event_id'] for r in guarded['trace']],[r['event_id'] for r in raw['trace']])
        request=next(e for e in p['events'] if e['kind']=='request')
        request['next']={'ALLOW':p['events'][0]['id']}
        with self.assertRaises(ValueError):validate_scenario(p)

    def test_reports_omit_input_text_but_reproducer_explicitly_contains_it(self):
        p=make_scenario();report=run_scenario(p)
        for event in p['events']:
            for key in ['text','value']:
                if key in event:self.assertNotIn(event[key],json.dumps(report,ensure_ascii=False))
        r=minimize_failure(p,lambda:ReferenceTarget('permissive'))
        self.assertTrue(r['text_in_report']);self.assertTrue(r['contains_synthetic_scenario_text'])

    def test_overblocking_does_not_score_as_good_utility(self):
        result=compare_reference_targets()
        self.assertEqual(result['results']['overblocking_enforced']['useful_task_preservation']['rate'],0)
        self.assertEqual(result['results']['overblocking_enforced']['unauthorized_deliveries']['count'],0)
        self.assertEqual(result['tradeoff']['eligible_under_declared_ceiling'],['guarded_enforced'])


class ConsistencyTests(unittest.TestCase):
    def test_agreement_can_be_wrong_and_abstention_is_not_a_pass(self):
        pair={'pair_id':'p','relation':'invariant','before':'a','after':'b','expected':['ALLOW','ALLOW']}
        report=evaluate_consistency([pair],lambda p:'BLOCK')
        self.assertTrue(report['pairs'][0]['relation_met']);self.assertFalse(report['pairs'][0]['reference_met'])
        self.assertEqual(report['failed'],1)
        self.assertEqual(evaluate_consistency([pair],lambda p:'REVIEW')['unresolved'],1)

    def test_required_transition_checks_direction(self):
        p={'pair_id':'p','relation':'transition','before':'before','after':'after','expected':['ALLOW','BLOCK']}
        self.assertEqual(evaluate_consistency([p],lambda v:'BLOCK' if v=='before' else 'ALLOW')['failed'],1)
        self.assertEqual(evaluate_consistency([p],lambda v:'ALLOW' if v=='before' else 'BLOCK')['passed'],1)
        p.pop('expected')
        with self.assertRaises(ValueError):evaluate_consistency([p],lambda v:'ALLOW')

    def test_reference_consistency_and_scope_changes(self):
        pairs=policy_pairs('uk')
        result=evaluate_consistency(pairs,lambda p:run_scenario(p)['requests'][-1]['outcome'])
        self.assertEqual(result['passed'],6)
        weak=evaluate_consistency(pairs,lambda p:run_scenario(p,lambda:ReferenceTarget('permissive'),enforce=False)['requests'][-1]['proposal'])
        self.assertEqual(weak['failed'],4)

    def test_invalid_frontier_rates_rejected(self):
        for bad in [float('nan'),True,-.1,1.1]:
            with self.assertRaises(ValueError):tradeoff_frontier([{'name':'x','unauthorized_rate':bad,'utility_rate':1}])

if __name__=='__main__':unittest.main()
