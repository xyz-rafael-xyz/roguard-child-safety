"""Verify the proposed sampler against study64 TRAIN identities, without a model.

Run with the repository on PYTHONPATH and explicit --root and --private paths.
The private directory supplies the retained composition54 TRAIN corpus and
CPU-only simulation. Reserved TEST input files are never read.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from training.response_sampling_v10 import MixedResponseSampler, STRATA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite sampler verification')
    evidence = args.root / 'research/v10-evidence'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    paths = {'curriculum25': evidence / 'response-curriculum-25/train.jsonl',
             'contrast32': evidence / 'contrast-response-32/additional-train.jsonl',
             'composition54': args.private / 'compositional-response-54/train.jsonl'}
    simulation_path = args.private / 'mixed-response-batch-simulation-64.json'
    simulation = json.loads(simulation_path.read_text())
    train = {}
    for source, path in paths.items():
        assert sha(path) == simulation['source_sha256'][source]
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['partition'] == 'train' and row['case_id'] not in train
            train[row['case_id']] = {**row, 'origin': 'generated' if source == 'composition54' else 'historical'}
    identity_path = evidence / 'warm-large-response-attention-64/feature-inputs.jsonl'
    identities = [json.loads(line) for line in identity_path.read_text().splitlines()]
    rows = [train[r['case_id']] if r['partition'] == 'train' else r for r in identities]
    indices = [i for i, r in enumerate(rows) if r['partition'] == 'train']
    assert len(rows) == 13476 and len(indices) == len(train) == 12884
    sampler = MixedResponseSampler(rows, indices)
    rng, replay = np.random.default_rng(1059), np.random.default_rng(1059)
    epochs = []
    for epoch in range(1, 7):
        batches = sampler.batches(rng)
        assert np.array_equal(batches, sampler.batches(replay))
        assert batches.shape == (3072, 8)
        visits = Counter(batches.ravel())
        assert set(visits) == set(indices)
        for i in indices:
            assert visits[i] in ({1} if rows[i]['origin'] == 'generated' else {20, 21})
        for batch in batches:
            assert len(set(batch)) == 8
            assert Counter((rows[i]['origin'], rows[i]['language']) for i in batch) == {key: 2 for key in STRATA}
        mass = sampler.weights[batches].mean(axis=1)
        cv = float(mass.std() / mass.mean())
        quantiles = dict(zip(('min', 'p05', 'median', 'p95', 'max'), np.quantile(mass, [0, .05, .5, .95, 1]).tolist()))
        expected = simulation['epochs'][epoch - 1]
        assert cv == expected['sample_weight_mass_coefficient_of_variation']
        assert quantiles == expected['mean_sample_weight_per_batch_quantiles']
        epoch_ids = [[rows[i]['case_id'] for i in batch] for batch in batches]
        epochs.append({'epoch': epoch, 'batches': len(batches), 'presentations': int(batches.size),
                       'ordered_case_ids_sha256': hashlib.sha256(json.dumps(epoch_ids, separators=(',', ':')).encode()).hexdigest(),
                       'all_train_rows_visited': True, 'all_generated_rows_visited_once': True,
                       'historical_repetitions': [20, 21], 'all_batches_have_two_per_stratum': True,
                       'duplicate_examples_within_batch': False, 'non_train_rows_used': False,
                       'seed_replay_exact': True, 'prior_simulation_statistics_exact': True,
                       'sample_weight_mass_coefficient_of_variation': cv,
                       'mean_sample_weight_per_batch_quantiles': quantiles})
    assert len({r['ordered_case_ids_sha256'] for r in epochs}) == 6
    assert all(sampler.weights[i] == 0 for i, r in enumerate(rows) if r['partition'] != 'train')
    report = {'complete': True, 'verification': 'PASS',
              'scope': 'TRAIN identities, coverage, repetition, batch composition, weights and seeded replay only; no encoder or gradient execution.',
              'sampler_source_sha256': sha(args.root / 'training/response_sampling_v10.py'),
              'verification_source_sha256': sha(Path(__file__)), 'source_train_sha256': {k: sha(p) for k, p in paths.items()},
              'identity_inventory_sha256': sha(identity_path), 'prior_simulation_sha256': sha(simulation_path),
              'train_rows': len(indices), 'batch_size': 8, 'epochs': epochs,
              'six_pass_presentations': sum(e['presentations'] for e in epochs),
              'gradient_variance_measured': False, 'training_launched': False, 'model_qualified': False,
              'reserved_test_opened': False, 'stable_release_ready': False}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'verification': 'PASS', 'epochs': len(epochs), 'train_rows': len(indices),
                      'presentations': report['six_pass_presentations'], 'training_launched': False}))


if __name__ == '__main__':
    main()
