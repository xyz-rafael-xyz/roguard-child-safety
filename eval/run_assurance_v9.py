"""Deterministic structural evaluation, not language-model accuracy."""
import argparse
import itertools
import json
from pathlib import Path
import random
import time

from roguard.assurance import FIELDS, ONTOLOGY, action, assure, digest, verify


def greedy_cost(worlds, costs):
    if len({action(world) for world in worlds}) == 1:
        return 0
    options = []
    for field, cost in costs.items():
        groups = [[w for w in worlds if w[field] == value] for value in sorted({w[field] for w in worlds})]
        if len(groups) > 1:
            ambiguity = max(len({action(w) for w in group}) for group in groups)
            options.append((ambiguity, cost, field, groups))
    _, cost, field, groups = min(options, key=lambda row: row[:3])
    remaining = {k:v for k,v in costs.items() if k != field}
    return cost + max(greedy_cost(group, remaining) for group in groups)


def run(count, seed):
    rng = random.Random(seed)
    universe = [dict(zip(FIELDS, values)) for values in itertools.product(*(ONTOLOGY[field] for field in FIELDS))]
    totals = dict(cases=count, world_paths_checked=0, leaf_errors=0, cost_errors=0,
                  worse_than_greedy=0, strictly_better_than_greedy=0,
                  optimal_cost_sum=0, greedy_cost_sum=0, max_states_evaluated=0)
    started = time.monotonic(); witness = None
    for index in range(count):
        worlds = rng.sample(universe, rng.randint(2,16))
        costs = {field:rng.randint(1,5) for field in FIELDS}
        problem = dict(language=('ro','uk')[index%2], hypotheses=worlds,
                       context_digest=digest({'seed':seed,'case':index}), question_costs=costs)
        report = assure(problem)
        verify(report)
        observed = 0
        for world in worlds:
            node = report['tree']; path_cost = 0
            while 'question' in node:
                field = node['question']; path_cost += costs[field]; node = node['branches'][world[field]]
            totals['leaf_errors'] += node['action'] != action(world)
            totals['world_paths_checked'] += 1
            observed = max(observed,path_cost)
        optimal = report['optimal_worst_case_cost']; greedy = greedy_cost(worlds,costs)
        totals['cost_errors'] += observed != optimal
        totals['worse_than_greedy'] += optimal > greedy
        totals['strictly_better_than_greedy'] += optimal < greedy
        totals['optimal_cost_sum'] += optimal; totals['greedy_cost_sum'] += greedy
        totals['max_states_evaluated'] = max(totals['max_states_evaluated'],report['states_evaluated'])
        if witness is None and optimal < greedy:
            witness = dict(problem=problem, optimal=optimal, greedy=greedy, certificate_digest=report['certificate_digest'])
    return dict(seed=seed, results=totals, seconds=time.monotonic()-started,
                improvement_example=witness, language_model_evaluated=False,
                generated_world_coverage_verified=False,
                baseline='Cost-aware one-step action-ambiguity greedy selection; deterministic field-name ties',
                limitation='Synthetic policy-world simulation with caller-supplied costs, not reviewer study or classifier accuracy.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=9001)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1 <= args.cases <= 100000:parser.error('cases must be 1..100000')
    report=run(args.cases,args.seed)
    with args.output.open('x',encoding='utf-8') as output:json.dump(report,output,indent=2)
    print(json.dumps(report['results']))
    if any(report['results'][key] for key in ('leaf_errors','cost_errors','worse_than_greedy')):
        raise SystemExit(1)


if __name__=='__main__':main()
