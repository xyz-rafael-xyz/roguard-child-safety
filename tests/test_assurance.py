from copy import deepcopy
import itertools
import random
import tempfile
from pathlib import Path
import unittest

from roguard.assurance import FIELDS, ONTOLOGY, action, answer, assure, assure_conversation, combine, digest, read_json, verify


def reference(world):
    if world['concern'] == '0':
        return 'CONTINUE'
    fiction = world['subject'] == 'FICTIONAL' and world['source'] == 'FICTION'
    conflict = (world['subject'] == 'FICTIONAL') != (world['source'] == 'FICTION')
    revision = world['assertion'] == 'RETRACTED' or world['relation'] == 'RETRACTS'
    if conflict or revision:
        return 'REVIEW_AND_CONTINUE'
    if fiction or world['assertion'] == 'NEGATED':
        return 'CONTINUE'
    if world['assertion'] == 'UNCERTAIN' or world['time'] == 'HYPOTHETICAL':
        return 'REVIEW_AND_CONTINUE'
    return 'SUPPORT_AND_REVIEW'


def brute_cost(worlds, costs):
    if len({reference(w) for w in worlds}) == 1:
        return 0
    options = []
    for field, cost in costs.items():
        values = {w[field] for w in worlds}
        if len(values) > 1:
            rest = {k: v for k, v in costs.items() if k != field}
            options.append(cost + max(brute_cost([w for w in worlds if w[field] == value], rest) for value in values))
    return min(options, default=float('inf'))


def problem(worlds, **kwargs):
    return dict(language='ro', hypotheses=worlds, context_digest=digest('synthetic context'), **kwargs)


def all_worlds():
    return [dict(zip(FIELDS, values)) for values in itertools.product(*(ONTOLOGY[f] for f in FIELDS))]


class AssuranceTests(unittest.TestCase):
    def test_complete_policy_and_tree_leaves(self):
        worlds = all_worlds()
        certificate = assure(problem(worlds))
        observed = 0
        for world in worlds:
            self.assertEqual(action(world), reference(world))
            node = certificate['tree']; cost = 0
            while 'question' in node:
                cost += node['cost']; node = node['branches'][world[node['question']]]
            self.assertEqual(node['action'], reference(world))
            observed = max(observed, cost)
        self.assertEqual(observed, certificate['optimal_worst_case_cost'])
        self.assertTrue(verify(certificate))

    def test_optimality_against_small_brute_force(self):
        rng = random.Random(901)
        worlds = all_worlds()
        for _ in range(40):
            chosen = rng.sample(worlds, 6)
            costs = {field: rng.randint(1, 5) for field in FIELDS}
            self.assertEqual(assure(problem(chosen, question_costs=costs))['optimal_worst_case_cost'], brute_cost(chosen, costs))

    def test_correlations_are_not_cartesianized(self):
        a = dict(subject='FICTIONAL', source='FICTION', concern='1', assertion='AFFIRMED', relation='NEW', time='CURRENT')
        b = dict(a, subject='SELF', source='DIRECT', concern='0')
        certificate = assure(problem([a,b]))
        self.assertEqual(certificate['possible_actions'], ['CONTINUE'])
        self.assertEqual(certificate['optimal_worst_case_cost'], 0)
        self.assertEqual(len(certificate['input']['hypotheses']), 2)

    def test_stale_tampered_and_outside_answers(self):
        worlds = all_worlds()[:3]
        certificate = assure(problem(worlds))
        changed = deepcopy(certificate); changed['permits_external_action'] = True
        with self.assertRaises(ValueError): verify(changed)
        changed = deepcopy(certificate); changed['optimal_worst_case_cost'] = False
        with self.assertRaises(ValueError): verify(changed)
        with self.assertRaises(ValueError): verify(certificate, expected_input_digest='0'*64)
        with self.assertRaises(ValueError): answer(certificate, {'subject':'SELF'}, expected_certificate_digest='0'*64)
        result = answer(certificate, {'subject':'OTHER'}, expected_certificate_digest=certificate['certificate_digest'])
        self.assertEqual(result['status'], 'OUTSIDE_HYPOTHESES')
        self.assertFalse(result['permits_external_action'])

    def test_answer_progress_and_unavailable_questions(self):
        worlds = all_worlds()
        certificate = assure(problem(worlds))
        field = certificate['tree']['question']; value = worlds[0][field]
        result = answer(certificate, {field:value}, expected_certificate_digest=certificate['certificate_digest'])
        self.assertEqual(result['status'], 'ANSWER_APPLIED')
        self.assertLess(len(result['certificate']['input']['hypotheses']), len(worlds))
        self.assertTrue(verify(result['certificate']))
        unresolved = assure(problem(worlds, question_costs={}))
        self.assertEqual(unresolved['status'], 'UNRESOLVED')
        self.assertIsNone(unresolved['optimal_worst_case_cost'])

    def test_invalid_inputs(self):
        base = problem(all_worlds()[:2])
        for update in ({'hypotheses': []}, {'language':'en'}, {'question_costs':{'concern':True}},
                       {'review_floor':1}, {'question_costs':{'concern':0}}, {'context_digest':'bad'},
                       {'hypotheses':[{}]}, {'hypotheses':all_worlds()*2}):
            with self.assertRaises(ValueError): assure(dict(base, **update))

    def test_permutation_and_duplicates(self):
        worlds = all_worlds()[::17]
        self.assertEqual(assure(problem(worlds)), assure(problem(list(reversed(worlds))+worlds[:2])))

    def test_ambiguous_json_and_resource_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.json'
            for content in ('{"language":"ro","language":"uk"}', '{"cost":NaN}', ' '*2_000_001):
                path.write_text(content,encoding='utf-8')
                with self.assertRaises(ValueError): read_json(path)

    def test_review_floor_cannot_clear(self):
        for world in all_worlds():
            certificate = assure(problem([world], review_floor=True))
            self.assertNotIn('CONTINUE', certificate['possible_actions'])

    def test_combination_retains_disagreement_and_provenance(self):
        a = dict(subject='SELF', source='DIRECT', concern='0', assertion='AFFIRMED', relation='NEW', time='CURRENT')
        b = dict(a, concern='1')
        first = assure(problem([a], model_digests=['1'*64]))
        second = assure(problem([b], model_digests=['2'*64]))
        combined = combine([first,second])
        self.assertEqual(combined['possible_actions'], ['CONTINUE','SUPPORT_AND_REVIEW'])
        self.assertEqual(combined['input']['model_digests'], ['1'*64,'2'*64])
        self.assertEqual(len(combined['input']['hypotheses']), 2)
        other = deepcopy(second['input']); other['context_digest'] = '0'*64
        with self.assertRaises(ValueError): combine([first,assure(other)])
        floor = assure(problem([a], review_floor=True))
        self.assertNotIn('CONTINUE',combine([first,floor])['possible_actions'])

    def test_actual_bilingual_runtime_bindings(self):
        for language, text in [('ro','Am nevoie de ajutor la matematică.'),('uk','Мені потрібна допомога з математикою.')]:
            payload = {'language':language,'turns':[{'role':'minor','text':text}]}
            report = assure_conversation(payload)
            self.assertTrue(report['entries'])
            self.assertNotIn(text, str(report))
            for entry in report['entries']:
                self.assertTrue(verify(entry['certificate']))
                self.assertFalse(entry['certificate']['permits_external_action'])
            changed = assure_conversation({'language':language,'turns':[{'role':'minor','text':text+'!'}]})
            self.assertNotEqual(report['context_digest'], changed['context_digest'])
