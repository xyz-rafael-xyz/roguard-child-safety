"""Independent saved global-VALUE pooling, token and CPU75-head audit.

No encoder, optimizer, accuracy labels, fitting or scale selection are used.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from importlib.metadata import version
from pathlib import Path

POOLS = ['focal', 'aligned_prior_minor', 'aligned_prior_nonminor',
         'chronological_prior1', 'chronological_prior2']
LOGITS = ['qualified75', 'qualified75_repeat']

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def finite(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Invalid nonnegative observation')
    return value

def module(path):
    spec = importlib.util.spec_from_file_location('bound135_' + Path(path).stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def derive_pools(record):
    roles = record['role_pool_positions']
    prior = sorted(roles[1] + roles[2])
    count = len(roles[1])
    return roles + [prior[:count], prior[count:]]

def pool_means(values, positions, np):
    require(values.dtype == np.float32 and values.ndim == 3 and
            values.shape[0] == 2 and values.shape[2] == 512 and
            387 < values.shape[1] <= 600 and np.isfinite(values).all(),
            'Saved global VALUE contract differs')
    require(len(positions) == 5, 'Five pools required')
    output = []
    # Independent head-major indexing and summation. No producer pooling helper.
    for indices in positions:
        require(indices == sorted(set(indices)) and
                all(type(i) is int and 387 <= i < values.shape[1] for i in indices),
                'Invalid global pool coordinates')
        if indices:
            selected = np.take(values, indices, axis=1).astype(np.float64)
            mean = (np.sum(selected, axis=1) / len(indices)).astype(np.float32)
            output.append(mean.reshape(1024))
        else:
            output.append(np.zeros(1024, np.float32))
    return np.stack(output)

def cpu_head(feature, weights, language, np, torch):
    with torch.no_grad():
        x = torch.from_numpy(feature.copy()).reshape(1, 2560)
        norm = (x * x).sum(1, keepdim=True).sqrt()
        require(bool(torch.isfinite(norm).all()) and bool((norm > 0).all()),
                'Invalid final-vector norm')
        x = x / norm * math.sqrt(2560)
        w = lambda name: torch.from_numpy(weights['heads.' + language + '.' + name])
        hidden = torch.nn.functional.linear(x, w('first.weight'), w('first.bias'))
        hidden = hidden * (1 + torch.erf(hidden / math.sqrt(2))) / 2
        result = torch.nn.functional.linear(hidden, w('last.weight'), w('last.bias')).numpy()[0].copy()
    require(result.shape == (19,) and result.dtype == np.float32 and np.isfinite(result).all(),
            'Invalid CPU75 projection')
    return result

def classes(z):
    cuts = (0, 3, 6, 10, 13, 15, 17, 19)
    return [int(z[a:b].argmax()) for a, b in zip(cuts[:-1], cuts[1:])]

def run(path, expected, out):
    require(sha(path) == expected, 'Audit plan changed')
    plan = read(path)
    require(plan['registered'] is True and plan['study'] == 135 and
            plan['producer_study'] == 134 and plan['encoder_permitted'] is False and
            type(plan['producer_exit_code']) is int and plan['producer_exit_code'] == 0,
            'Unregistered or nonterminal saved audit')
    require(sha(__file__) == plan['source_sha256'], 'Auditor changed')
    source_raw, plan_raw = Path(__file__).read_bytes(), Path(path).read_bytes()
    require(hashlib.sha256(source_raw).hexdigest() == plan['source_sha256'] and
            hashlib.sha256(plan_raw).hexdigest() == expected, 'Audit snapshot changed')
    def check():
        require(sha(path) == expected and sha(__file__) == plan['source_sha256'], 'Audit identity changed')
        for key, binding in plan['files'].items():
            require(sha(binding['path']) == binding['sha256'], 'Audit input changed: ' + key)
    check()
    for key, value in plan['versions'].items():
        require(version(key) == value, 'Numerical version changed')
    f = lambda key: Path(plan['files'][key]['path'])
    producer = module(f('producer_source'))
    parent, paths, inventory, records = producer.preflight(
        f('producer_plan'), plan['files']['producer_plan']['sha256'])
    report = read(f('producer_report'))
    root = f('producer_report').parent
    require(report['complete'] is True and report['qualified'] is False and
            report['training_performed'] is False and report['accuracy_evaluation_performed'] is False and
            report['full_hidden_states_saved'] is False and report['normalized_global_values_saved'] is True and
            report['learned_residual_implemented'] is False, 'Producer scope incomplete or changed')
    require(report['source_sha256'] == sha(f('producer_source')) and
            report['plan_sha256'] == sha(f('producer_plan')), 'Producer lineage changed')
    require(report['pool_names'] == POOLS and report['logit_names'] == LOGITS,
            'Producer feature schema differs')
    expected_names = {'source.py', 'registered-plan.json', 'inventory.jsonl', 'alignment-records.jsonl',
                      'qualified-final.npy', 'value-pools.npy', 'metadata.npy', 'logits.npy'}
    expected_names |= {row['case_id'] + suffix for row in inventory
                       for suffix in ('-observation.json', '-global-value23.npy')}
    require(len(expected_names) == 40 and set(report['artifacts']) == expected_names,
            'Exact40 artifact inventory differs')
    snapshot = {name: sha(root / name) for name in expected_names}
    require(snapshot == report['artifacts'], 'Artifact bytes changed')
    def barrier():
        check()
        producer.bindings(parent)
        producer.origins(parent)
        require({name: sha(root / name) for name in expected_names} == snapshot, 'Saved artifacts changed')
        require({p.name for p in root.iterdir()} == expected_names | {'report.json'},
                'Saved inventory or failure marker differs')
    barrier()
    require(sum((root / name).stat().st_size for name in expected_names) == report['non_report_artifact_bytes'],
            'Artifact byte accounting differs')
    require(sum(p.stat().st_size for p in root.iterdir()) <= 64 * 1024**2, 'Output cap exceeded')
    for name, bound in [('source.py', f('producer_source')), ('registered-plan.json', f('producer_plan')),
                        ('inventory.jsonl', paths['inventory']), ('alignment-records.jsonl', paths['alignments'])]:
        require((root / name).read_bytes() == bound.read_bytes(), 'Snapshot differs')
    import numpy as np
    import torch
    from safetensors.numpy import load_file
    from roguard.gemma4_review import _score
    producer.origins(parent)
    arrays = {}
    for name, shape in [('qualified-final.npy', (16, 2560)), ('value-pools.npy', (16, 5, 1024)),
                        ('metadata.npy', (16, 6)), ('logits.npy', (16, 2, 19))]:
        value = np.load(root / name, allow_pickle=False)
        require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                'Saved aggregate array contract differs')
        arrays[name] = value
    alignment = module(paths['alignment_source'])
    barrier()
    token_rows = 0
    with alignment.pinned_tokenizer(parent['base_snapshot']) as (tokenizer, provenance):
        for row, record in zip(inventory, records):
            actual = producer.enrich(alignment.align(row['payload'], tokenizer, row.get('focus')), row)
            require(canonical(actual) == canonical(record), 'Independent retokenization differs')
            token_rows += 1
    barrier()
    results = {}
    cpu_outputs = np.empty((16, 19), np.float32)
    cpu_pools = np.empty((16, 5, 1024), np.float32)
    for language in ('ro', 'uk'):
        info = report['languages'][language]
        require(info['complete'] is True and type(info['checked']) is int and info['checked'] == 8 and
                len(info['observations']) == 8 and not info['cleanup_failures'], 'Incomplete language')
        for key in ('load_seconds', 'head_install_seconds', 'prefix_seconds'):
            finite(info[key])
        sampling = info['disk_sampling']
        samples = sampling['samples']
        require(isinstance(samples, list) and samples, 'Missing disk samples')
        for sample in samples:
            finite(sample['seconds'])
            for key in ('output_free', 'temp_free'):
                require(type(sample[key]) is int and sample[key] >= 8 * 1024**3, 'Disk reserve failed')
        require(all(a['seconds'] <= b['seconds'] for a, b in zip(samples, samples[1:])), 'Sample order differs')
        require(min(v['output_free'] for v in samples) == sampling['minimum_output_free'] and
                min(v['temp_free'] for v in samples) == sampling['minimum_temp_free'], 'Disk minima differ')
        weights, spec = load_file(str(paths[language + '_parent_weights'])), read(paths[language + '_spec'])
        max_logit = max_score = max_pool = 0.0
        mask_differences = 0
        primary_times, baseline_times = [], []
        selected = [i for i, row in enumerate(inventory) if row['payload']['language'] == language]
        require(len(selected) == 8, 'Language membership differs')
        for ordinal, i in enumerate(selected):
            row, record = inventory[i], records[i]
            observation = read(root / (row['case_id'] + '-observation.json'))
            require(canonical(observation) == canonical(info['observations'][ordinal]), 'Observation differs')
            require(observation['case_id'] == row['case_id'] and observation['language'] == language and
                    observation['token_sha256'] == record['classifier_token_sha256'] and
                    observation['token_count'] == record['classifier_token_count'], 'Observation identity differs')
            require(observation['pool_names'] == POOLS and observation['logit_names'] == LOGITS,
                    'Observation schema differs')
            positions = derive_pools(record)
            require(observation['global_positions'] == positions and
                    observation['pool_counts'] == list(map(len, positions)) and
                    observation['pool_validity'] == [bool(p) for p in positions], 'Pool membership differs')
            same = positions[1:3] == positions[3:]
            require(observation['chronological_masks_identical'] is same, 'Chronological identity differs')
            mask_differences += not same
            meta = np.asarray([len(p) / 1024 for p in positions[:3]] +
                              list(map(float, record['shared_structural_role_presence'])), dtype=np.float32)
            require(arrays['metadata.npy'][i].tobytes() == meta.tobytes() and
                    observation['shared_metadata'] == meta.tolist() and
                    observation['role_presence'] == record['shared_structural_role_presence'], 'Metadata differs')
            value_name = row['case_id'] + '-global-value23.npy'
            require(observation['global_value_file'] == value_name and
                    observation['global_value_sha256'] == snapshot[value_name] and
                    observation['global_value_layout'] == 'H,L,D', 'Saved VALUE identity differs')
            values = np.load(root / value_name, allow_pickle=False)
            require(values.shape == (2, record['classifier_token_count'], 512) and
                    observation['global_value_shape'] == list(values.shape), 'VALUE length differs')
            means = pool_means(values, positions, np)
            errors = np.max(np.abs(arrays['value-pools.npy'][i].astype(np.float64) - means.astype(np.float64)), axis=1)
            require(bool((errors <= 1e-5).all()) and observation['pool_mean_max_errors'] == errors.tolist(),
                    'Independent saved pool means differ')
            for j, positions_j in enumerate(positions):
                if not positions_j:
                    require(arrays['value-pools.npy'][i, j].tobytes() == np.zeros(1024, np.float32).tobytes(),
                            'Empty pool not exact positive zero')
            max_pool = max(max_pool, float(errors.max()))
            cpu_pools[i] = means
            z, repeat = arrays['logits.npy'][i]
            require(z.tobytes() == repeat.tobytes() and observation['qualified_repeat_logit_error'] == 0 and
                    observation['qualified_repeat_feature_error'] == 0, 'Qualified repeat differs')
            for key in ('qualified_seven_argmax_differences', 'qualified_binary_differences'):
                require(type(observation[key]) is int and observation[key] == 0, 'Parity count differs')
            require(observation['zero_residual_same_object'] is True and
                    observation['zero_residual_bytes_identical'] is True, 'Identity observation differs')
            cpu = cpu_head(arrays['qualified-final.npy'][i], weights, language, np, torch)
            logit_error = float(np.max(np.abs(cpu.astype(np.float64) - z.astype(np.float64))))
            require(logit_error <= 1e-3 and classes(cpu) == classes(z), 'CPU75 head differs')
            score, cpu_score = float(_score(z, spec)), float(_score(cpu, spec))
            require(score == observation['score'] and
                    abs(score - observation['served_score']) == observation['score_parity_error'] and
                    observation['score_parity_error'] <= 1e-10, 'Native serving score observation differs')
            require((cpu_score >= spec['threshold']) == (score >= spec['threshold']) ==
                    (observation['served_score'] >= spec['threshold']), 'CPU75 policy decision differs')
            cpu_outputs[i] = cpu
            max_logit, max_score = max(max_logit, logit_error), max(max_score, abs(cpu_score - score))
            timing = observation['pool_phase_times']
            require(set(timing) == {'native_value_dtype', 'retained_value_dtype', 'native_value_shape',
                                   'value_float32_conversion_seconds', 'native_pool_seconds', 'value_host_copy_seconds'},
                    'Pool phase schema differs')
            require(timing['retained_value_dtype'] == 'float32' and
                    timing['native_value_dtype'] in ('float32', 'float16', 'bfloat16',
                                                     'mlx.core.float32', 'mlx.core.float16', 'mlx.core.bfloat16') and
                    timing['native_value_shape'] == [1, 2, record['classifier_token_count'], 512],
                    'VALUE conversion observation differs')
            for key in ('value_float32_conversion_seconds', 'native_pool_seconds', 'value_host_copy_seconds'):
                finite(timing[key])
            for key in ('qualified_baseline_seconds', 'primary_single_path_seconds',
                        'saved_value_cpu_mean_audit_seconds', 'value_persistence_readback_seconds',
                        'aggregate_persistence_seconds', 'extra_qualified_repeat_seconds',
                        'extra_runtime_reference_seconds', 'seconds'):
                finite(observation[key])
            primary = (observation['qualified_baseline_seconds'] + timing['value_float32_conversion_seconds'] +
                       timing['native_pool_seconds'])
            require(primary == observation['primary_single_path_seconds'] and primary <= observation['seconds'],
                    'Primary phase arithmetic differs')
            for key in ('process_peak_rss_bytes', 'process_current_rss_bytes'):
                require(type(observation[key]) is int and 0 < observation[key] <= 12 * 1024**3, 'RSS invalid')
            for key in ('mlx_active_memory', 'mlx_peak_memory', 'mlx_cache_memory'):
                require(type(observation[key]) is int and observation[key] >= 0, 'MLX observation invalid')
            require(type(observation['disk_free']) is int and observation['disk_free'] >= 8 * 1024**3,
                    'Observed reserve failed')
            primary_times.append(primary)
            baseline_times.append(observation['qualified_baseline_seconds'])
        require(mask_differences == 3, 'Expected role/chronological split differs')
        require(info['load_seconds'] + info['head_install_seconds'] + info['prefix_seconds'] +
                sum(v['seconds'] for v in info['observations']) <= 900, 'Recorded time lower bound exceeds limit')
        results[language] = dict(rows=8, pools_independently_recomputed=40,
            maximum_pool_mean_error=max_pool, maximum_cpu75_logit_error=max_logit,
            maximum_cpu75_score_difference=max_score, seven_argmax_differences=0,
            cpu_binary_decision_differences=0, chronological_masks_different=mask_differences,
            primary_single_path_seconds=primary_times, qualified_baseline_seconds=baseline_times)
    barrier()
    out = Path(out)
    require(not out.exists(), 'Refusing overwrite')
    out.mkdir()
    (out / 'source.py').write_bytes(source_raw)
    (out / 'registered-plan.json').write_bytes(plan_raw)
    for name, values in [('cpu75-logits.npy', cpu_outputs), ('cpu-value-pools.npy', cpu_pools)]:
        np.save(out / name, values, allow_pickle=False)
        actual = np.load(out / name, allow_pickle=False)
        require(actual.dtype == np.float32 and actual.shape == values.shape and
                np.isfinite(actual).all() and actual.tobytes() == values.tobytes(), 'Audit array readback differs')
    require(sha(out / 'source.py') == plan['source_sha256'] and sha(out / 'registered-plan.json') == expected,
            'Audit snapshot differs')
    result = dict(verification='PASS', study=135, source_sha256=sha(__file__), plan_sha256=expected,
        producer_report_sha256=sha(f('producer_report')), token_rows_replayed=token_rows, head_rows_replayed=16,
        saved_value_arrays_checked=16, pooled_vectors_independently_recomputed=80, languages=results,
        encoder_loaded=False, training_performed=False, qualified=False,
        limitations=['Independent retokenization, float64 saved-VALUE means and CPU75 heads; no encoder replay.',
                    'Native VALUE origin, repeated feature and resource measurements are producer observations.',
                    'CPU score differences are reported separately and do not claim native 1e-10 parity.',
                    'Time reconstruction is a lower bound. Phase timings exclude load, prefix and audit/persistence overhead.',
                    'Zero-residual same-object and installation proofs are producer observations; no learned residual exists.'])
    result['artifacts'] = {name: sha(out / name) for name in
                           ('source.py', 'registered-plan.json', 'cpu75-logits.npy', 'cpu-value-pools.npy')}
    try:
        barrier()
        (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        require(canonical(read(out / 'report.json')) == canonical(result), 'Audit report readback differs')
        report_sha = sha(out / 'report.json')
        barrier()
        require({name: sha(out / name) for name in result['artifacts']} == result['artifacts'] and
                sha(out / 'report.json') == report_sha, 'Closing audit evidence changed')
    except BaseException as error:
        (out / 'report.json').write_text(json.dumps(dict(verification='FAIL', qualified=False, reason=str(error))) + '\n')
        raise
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.plan_sha256, args.output)
    print(json.dumps({key: result[key] for key in ('verification', 'token_rows_replayed',
                                                  'head_rows_replayed', 'pooled_vectors_independently_recomputed')}))
