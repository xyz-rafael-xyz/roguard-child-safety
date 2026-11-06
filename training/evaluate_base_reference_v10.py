"""Fresh base-view inference for a frozen reference-head candidate; no fitting."""
import argparse
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_attention import capture, capture_with_prefix, replay
from roguard.gemma4_contract import classifier_tokens, PREFIX_TOKENS
from roguard.gemma4_review import _Engine, _score
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl, prefix, input_identity
from roguard.trajectory_evaluation import evaluate, paired_compare, checkpoint_id
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory
from training.base_reference_head_v10 import make_head
from training.extract_base_reference_v10 import without_adapters
from training.evaluate_semantic_readout_v10 import world_from_logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'candidate', 'selected', 'selected_study', 'reference_study',
                'reference_private', 'selector', 'study', 'private'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    p.add_argument('--specialists', type=Path)
    p.add_argument('--attention-study', type=Path)
    p.add_argument('--attention-regression', type=Path)
    p.add_argument('--paired-cache-study', type=Path)
    p.add_argument('--paired-cache-private', type=Path)
    p.add_argument('--query', action='store_true')
    a = p.parse_args()
    if a.query and (a.specialists is not None or a.paired_cache_study is not None or a.paired_cache_private is not None):
        raise ValueError('Query regression requires fresh query inference without old-view specialists or paired caches')
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite base-reference evaluation')
    selection = json.loads((a.selected_study/'report.json').read_text())
    selected_protocol = json.loads((a.selected_study/'protocol.json').read_text())
    reference = json.loads((a.reference_study/'report.json').read_text())
    ref_protocol = json.loads((a.reference_study/'protocol.json').read_text())
    if (not selection['complete'] or not reference['complete']
            or ref_protocol['format'] != 'roguard-v10-semantic-regression-1'
            or selected_protocol['format'] != ('roguard-query-reference-fit-1' if a.query else 'roguard-base-reference-fit-1')):
        raise ValueError('Expected complete, compatible parent studies')
    if selected_protocol['head_source_sha256'] != file_sha('training/base_reference_head_v10.py'):
        raise ValueError('Classifier implementation differs')
    if a.query and selected_protocol['query_helper_sha256'] != file_sha('training/probe_assertion_query_v10.py'):
        raise ValueError('Query tokenization implementation differs')
    for lang, artifacts in reference['features'].items():
        for suffix, expected in artifacts.items():
            if file_sha(a.reference_private/(lang+'-'+suffix)) != expected:
                raise ValueError('Original reference binding differs')
    paths = {'broad': Path('research/v10-evidence/broad-v9-regression/cases.jsonl'),
             **{name: Path('research/v10-evidence/data')/(name+'.jsonl')
                for name in ('dev', 'challenge', 'v9-regression')}}
    for name, path in paths.items():
        if file_sha(path) != ref_protocol['inputs'][name]:
            raise ValueError('Evaluation population differs')
    selector = TrajectoryModel(a.selector)
    if selector.identity != ref_protocol['selector_sha256']:
        raise ValueError('Memory selection differs')
    specialists = None
    if a.specialists is not None:
        if a.attention_study is None or a.attention_regression is None:
            raise ValueError('Specialists require bound attention studies')
        specialists = json.loads((a.specialists/'report.json').read_text())
        if (specialists['format'] != 'roguard-language-specialists-selection-1'
                or specialists['parent_report_sha256']['reference'] != file_sha(a.selected_study/'report.json')
                or specialists['parent_report_sha256']['attention'] != file_sha(a.attention_study/'report.json')):
            raise ValueError('Specialist parent binding differs')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    protocol = {'format': 'roguard-query-reference-regression-1' if a.query else 'roguard-base-reference-regression-1', 'source_sha256': file_sha(Path(__file__)),
        'head_source_sha256': file_sha('training/base_reference_head_v10.py'),
        'extractor_source_sha256': file_sha('training/extract_base_reference_v10.py'),
        'selection_report_sha256': file_sha(a.selected_study/'report.json'),
        'reference_report_sha256': file_sha(a.reference_study/'report.json'),
        'inputs': {name: file_sha(path) for name, path in paths.items()}, 'selector_sha256': selector.identity,
        'selected_heads': {lang: file_sha(a.selected/(lang+'-selected.safetensors')) for lang in ('ro', 'uk')
                           if specialists is None or specialists['languages'][lang]['selected'] == 'reference'},
        'fresh_zero_adapter_base_inference': True, 'published_v9_features_reused': True,
        'end_to_end_latency_measured': False, 'fitting_performed': False, 'threshold_selection_performed': False,
        'independent_test': False, 'stable_release_ready': False}
    if a.query:
        protocol['query_prompt'] = selected_protocol['query_prompt']
        protocol['query_inference_sha256'] = file_sha('training/query_inference_v10.py')
    if specialists is not None:
        protocol['specialist_selection_sha256'] = file_sha(a.specialists/'report.json')
        protocol['reused_attention_regression_sha256'] = file_sha(a.attention_regression/'report.json')
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file
    cases = {name: read_jsonl(path) for name, path in paths.items()}
    baseline = {name: read_jsonl(a.reference_study/(name+'-baseline.jsonl')) for name in paths}
    baseline_by_id = {name: {r['checkpoint_id']: r for r in rows} for name, rows in baseline.items()}
    outputs = {name: [] for name in paths}; started = time.monotonic(); done = 0; forwards = 0
    maximum_feature_error = 0.; maximum_live_score_error = 0.; checked = []; fresh_languages = []; reused_features = 0
    maximum_query_parity_error = 0.
    paired_cache = None
    if a.paired_cache_study is not None:
        if a.paired_cache_private is None:
            raise ValueError('Paired feature cache directory required')
        paired_cache = json.loads((a.paired_cache_study/'report.json').read_text())
        cache_protocol = json.loads((a.paired_cache_study/'protocol.json').read_text())
        if (not paired_cache['complete'] or cache_protocol['format'] != 'roguard-base-reference-regression-1'
                or cache_protocol['extractor_source_sha256'] != file_sha('training/extract_base_reference_v10.py')
                or cache_protocol['reference_report_sha256'] != file_sha(a.reference_study/'report.json')
                or cache_protocol['selector_sha256'] != selector.identity
                or cache_protocol['inputs'] != protocol['inputs']):
            raise ValueError('Paired cache representation or population differs')
        for lang, artifacts in paired_cache['features'].items():
            if file_sha(a.paired_cache_study/'v9-binding'/(lang+'-review.json')) != frozen.models[lang].identity:
                raise ValueError('Paired cache v9 encoder binding differs')
            for suffix, expected_sha in artifacts.items():
                if file_sha(a.paired_cache_private/(lang+'-'+suffix)) != expected_sha:
                    raise ValueError('Paired feature cache bytes differ')
        protocol['paired_cache_report_sha256'] = file_sha(a.paired_cache_study/'report.json')
        protocol['paired_cache_languages'] = sorted(paired_cache['features'])
        (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for lang in ('ro', 'uk'):
        if specialists is not None and specialists['languages'][lang]['selected'] == 'attention':
            previous = json.loads((a.attention_regression/'report.json').read_text())
            previous_protocol = json.loads((a.attention_regression/'protocol.json').read_text())
            bound = specialists['languages'][lang]
            if (not previous['complete'] or previous_protocol['format'] != 'roguard-semantic-tail-regression-1'
                    or previous_protocol['selection_report_sha256'] != file_sha(a.attention_study/'report.json')
                    or previous_protocol['selected_adapters'][lang] != bound['artifact_sha256']
                    or previous_protocol['selector_sha256'] != selector.identity
                    or bound['spec_sha256'] != file_sha(a.attention_study/(lang+'-review.json'))):
                raise ValueError('Reused attention pipeline binding differs')
            attention_spec = json.loads((a.attention_study/(lang+'-review.json')).read_text())
            for name, path in paths.items():
                prediction_path = a.attention_regression/(name+'-candidate.jsonl')
                if previous_protocol['inputs'][name] != file_sha(path) or previous['prediction_sha256'][name] != file_sha(prediction_path):
                    raise ValueError('Reused attention population or predictions differ')
                expected_inputs = {checkpoint_id(c, point): input_identity(prefix(c, point['focus']))
                                   for c in cases[name] if c['language'] == lang for point in c['checkpoints']}
                reused = [r for r in read_jsonl(prediction_path) if r['language'] == lang]
                if len(reused) != len(expected_inputs) or {r['checkpoint_id'] for r in reused} != set(expected_inputs):
                    raise ValueError('Reused attention predictions are incomplete')
                for row in reused:
                    if row['input_digest'] != expected_inputs[row['checkpoint_id']]:
                        raise ValueError('Reused attention input differs')
                    score = _score(row['field_logits'], attention_spec)
                    if abs(score-row['score']) > 1e-7 or int(score >= attention_spec['threshold']) != row['predicted']:
                        raise ValueError('Reused attention decision differs')
                outputs[name].extend(reused); done += len(reused)
            continue
        spec = json.loads((a.selected_study/(lang+'-head.json')).read_text())
        if spec['format'] != ('roguard-query-reference-head-1' if a.query else 'roguard-base-reference-head-1'):
            raise ValueError('Selected representation format differs')
        selected_record = selection['languages'][lang]['selected']
        if (any(spec[k] != selected_record[k] for k in ('mode', 'threshold', 'head_sha256'))
                or spec['configuration']['name'] != selected_record['candidate']):
            raise ValueError('Selected classifier metadata differs from the frozen development choice')
        if a.query and (spec['query_prompt'] != selected_protocol['query_prompt']
                        or spec['feature_report_sha256'] != selected_protocol['feature_report_sha256']):
            raise ValueError('Query head and fitted representation differ')
        if spec['head_sha256'] != protocol['selected_heads'][lang] or spec['v9_identity'] != frozen.models[lang].identity:
            raise ValueError('Selected head binding differs')
        if specialists is not None and specialists['languages'][lang]['spec_sha256'] != file_sha(a.selected_study/(lang+'-head.json')):
            raise ValueError('Specialist reference metadata differs')
        fresh_languages.append(lang)
        cached = paired_cache is not None and lang in paired_cache['features']
        engine = None if cached else _Engine(*frozen.models[lang]._args)
        query_view = None
        if a.query:
            from training.query_inference_v10 import QueryView
            query_view = QueryView(engine, selected_protocol['query_prompt'])
        head = make_head(spec['configuration'])
        tensors = load_file(a.selected/(lang+'-selected.safetensors'))
        expected = dict(tree_flatten(head.parameters()))
        if set(tensors) != set(expected) or any(tensors[k].shape != v.shape or not np.isfinite(tensors[k]).all() for k, v in expected.items()):
            raise ValueError('Selected classifier tensor inventory differs')
        head.load_weights([(k, mx.array(v)) for k, v in tensors.items()]); head.eval(); mx.eval(head.parameters())
        common = None
        if not cached:
            common = engine.make_cache(engine.model.base)
            mx.eval(engine.model.base.model(mx.array([PREFIX_TOKENS]), cache=common))
        mappings = read_jsonl(a.reference_private/(lang+'-rows.jsonl'))
        lookup = {(r['population'], r['checkpoint_id']): i for i, r in enumerate(mappings)}
        old_h = np.load(a.reference_private/(lang+'-features.npy'), mmap_mode='r')
        old_z = np.load(a.reference_private/(lang+'-original-logits.npy'), mmap_mode='r')
        records = [(name, c, point) for name, rows in cases.items() for c in rows if c['language'] == lang for point in c['checkpoints']]
        cached_values = None; cached_mapping = {}
        if cached:
            cached_rows = read_jsonl(a.paired_cache_private/(lang+'-rows.jsonl'))
            cached_values = np.load(a.paired_cache_private/(lang+'-features.npy'), mmap_mode='r')
            cached_mapping = {(r['population'], r['checkpoint_id']): (i, r) for i, r in enumerate(cached_rows)}
            if (cached_values.shape != (len(records), 5120) or len(cached_mapping) != len(records)
                    or set(cached_mapping) != {(name, checkpoint_id(c, point)) for name, c, point in records}):
                raise ValueError('Paired cache checkpoint inventory differs')
        features = np.lib.format.open_memmap(a.private/(lang+'-features.npy'), mode='w+', dtype=np.float32, shape=(len(records), 5120))
        new_logits = np.empty((len(records), 19), dtype=np.float32)
        new_mappings = []; memo = {}; checked_populations = set()
        identity = digest({'spec': spec, 'selector': selector.identity, 'budget': 1200,
                           'representation': 'published-v9-plus-semantic-query-v1' if a.query else 'published-v9-plus-zero-adapter-base-v1'})
        for i, (population, case, point) in enumerate(records):
            key = checkpoint_id(case, point); old = baseline_by_id[population][key]
            payload = prefix(case, point['focus']); view, inference = prepare_memory(payload, selector, 1200, preserve_complete=True)
            inference_digest = input_identity(inference)
            if view['mandatory_overflow'] or len(inference['turns']) > 12:
                raise ValueError('Unsupported memory input')
            position = lookup[(population, key)]; mapping = mappings[position]
            if any(r['input_digest'] != input_identity(payload) or r['inference_input_digest'] != inference_digest for r in (old, mapping)):
                raise ValueError('Original reference input binding differs')
            old_score = _score(old_z[position], frozen.models[lang]._spec)
            if abs(old_score-old['score']) > 1e-7 or int(old_score >= frozen.models[lang]._spec['threshold']) != old['predicted']:
                raise ValueError('Original baseline score binding differs')
            if cached:
                cache_position, cache_row = cached_mapping[(population, key)]
                if cache_row['input_digest'] != input_identity(payload) or cache_row['inference_input_digest'] != inference_digest:
                    raise ValueError('Paired cache actual input differs')
                paired = cached_values[cache_position].copy()
                z = head(mx.array(paired[None, :])); mx.eval(z)
                values = np.asarray(z)[0].copy(); score = _score(values, spec)
                reused_features += 1
            elif inference_digest not in memo:
                tokens = classifier_tokens(engine.tokenizer, inference)
                if len(tokens) > 1024:
                    raise ValueError('Model token limit exceeded')
                state = None
                if not a.query or population not in checked_populations:
                    if len(tokens) > len(PREFIX_TOKENS) and tokens[:len(PREFIX_TOKENS)] == PREFIX_TOKENS:
                        state = capture_with_prefix(engine.model.base, mx.array([tokens[len(PREFIX_TOKENS):]]), common, start=24)
                    else:
                        state = capture(engine.model.base, mx.array([tokens]), start=24)
                live_adapted = None
                if population not in checked_populations:
                    live_adapted = replay(engine.model.base, state)[:, 0, :]; mx.eval(live_adapted)
                    difference = float(np.max(np.abs(np.asarray(live_adapted)[0]-old_h[position])))
                    maximum_feature_error = max(maximum_feature_error, difference)
                    if difference > 1e-3:
                        raise ValueError('Reused adapted representation differs from live replay')
                if a.query:
                    base_array = query_view(inference, verify_full=population not in checked_populations)
                    maximum_query_parity_error = max(maximum_query_parity_error, query_view.maximum_full_parity_error)
                else:
                    with without_adapters(engine.model.base):
                        base_h = replay(engine.model.base, state)[:, 0, :]; mx.eval(base_h)
                        base_array = np.asarray(base_h)[0].copy()
                        del base_h
                paired = np.concatenate((old_h[position], base_array))
                z = head(mx.array(paired[None, :])); mx.eval(z)
                values = np.asarray(z)[0].copy(); score = _score(values, spec)
                if live_adapted is not None:
                    live = head(mx.concatenate((live_adapted, mx.array(base_array[None, :])), axis=1)); mx.eval(live)
                    live_score = _score(np.asarray(live)[0], spec)
                    maximum_live_score_error = max(maximum_live_score_error, abs(live_score-score))
                    if abs(live_score-score) > 1e-5 or int(live_score >= spec['threshold']) != int(score >= spec['threshold']):
                        raise ValueError('Reused and live paired classifier decisions differ')
                    checked_populations.add(population); checked.append(lang+':'+population)
                    del live, live_adapted
                memo[inference_digest] = paired, values, score; forwards += 1
                del state, z
            if not cached:
                paired, values, score = memo[inference_digest]
            features[i] = paired; new_logits[i] = values
            new_mappings.append({'population': population, 'checkpoint_id': key,
                'input_digest': input_identity(payload), 'inference_input_digest': inference_digest})
            outputs[population].append({**old, 'model_identity': identity, 'score': score,
                'predicted': int(score >= spec['threshold']), 'field_logits': values.tolist(),
                'interpretation_set': {'hypotheses': [world_from_logits(values)],
                    'reason': 'UNCALIBRATED_ARGMAX_DIAGNOSTIC', 'coverage_assumptions_verified': False,
                    'outside_ontology_ruled_out': False}})
            done += 1
            if done % 100 == 0:
                features.flush()
                progress = {'done': done, 'fresh_base_forwards': forwards, 'language': lang,
                            'population': population, 'seconds': time.monotonic()-started}
                (a.study/'progress.json').write_text(json.dumps(progress, indent=2)+'\n'); print(json.dumps(progress), flush=True)
            mx.clear_cache()
        features.flush(); del features
        np.save(a.private/(lang+'-candidate-logits.npy'), new_logits, allow_pickle=False)
        write_jsonl(a.private/(lang+'-rows.jsonl'), new_mappings)
        for name, rows in outputs.items():
            write_jsonl(a.study/(name+'-candidate.jsonl'), rows)
        del engine, query_view, head, common, memo, old_h, old_z, new_logits, cached_values
        import gc
        gc.collect(); mx.clear_cache()
    report = {'complete': True, 'populations': {}, 'fresh_base_forwards': forwards,
              'paired_feature_rows_reused': reused_features,
              'maximum_adapted_feature_replay_error': maximum_feature_error,
              'maximum_live_paired_score_error': maximum_live_score_error, 'live_checked_populations': checked,
              'maximum_query_full_parity_error': maximum_query_parity_error if a.query else None,
              'seconds': time.monotonic()-started, 'fitting_performed': False,
              'independent_test': False, 'stable_release_ready': False,
              'features': {lang: {suffix: file_sha(a.private/(lang+'-'+suffix)) for suffix in
                            ('features.npy', 'candidate-logits.npy', 'rows.jsonl')} for lang in fresh_languages}}
    if specialists is not None:
        report['language_specialists'] = specialists['languages']
    for name, rows in cases.items():
        write_jsonl(a.study/(name+'-candidate.jsonl'), outputs[name])
        report['populations'][name] = {'baseline': evaluate(rows, baseline[name]),
            'candidate': evaluate(rows, outputs[name]), 'comparison': paired_compare(rows, baseline[name], outputs[name])}
    report['prediction_sha256'] = {name: file_sha(a.study/(name+'-candidate.jsonl')) for name in cases}
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name: {lang: g['strict_accuracy'] for lang, g in r['candidate']['groups'].items()}
                      for name, r in report['populations'].items()}, indent=2))


if __name__ == '__main__':
    main()
