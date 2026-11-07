"""Locate cached/full-forward divergence without changing weights or thresholds."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
import time

from roguard.trajectory_data import file_sha


def selected_cases(observations):
    """First DEV control plus two largest DEV discrepancies per path/metric."""
    rows = [r for r in observations if r['partition'] == 'dev']
    if not rows or len({r['feature_index'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate development observations')
    chosen = {rows[0]['feature_index']}
    for name in ('prefix', 'uncached'):
        for metric in ('feature_error', 'logit_error', 'full_replay_error'):
            available = [r for r in rows if metric in r['paths'][name]]
            ranked = sorted(available, key=lambda r: (-r['paths'][name][metric], r['feature_index']))
            chosen.update(r['feature_index'] for r in ranked[:2])
    return [r for r in rows if r['feature_index'] in chosen]


@contextmanager
def observe_layers(layers, callback):
    """Observe native layer results; restore the class even after inference errors."""
    cls = type(layers[0])
    if any(type(layer) is not cls for layer in layers):
        raise ValueError('Mixed decoder layer types')
    original = cls.__call__
    indices = {id(layer): i for i, layer in enumerate(layers)}

    def observed(layer, *args, **kwargs):
        result = original(layer, *args, **kwargs)
        if id(layer) in indices:
            callback(indices[id(layer)], result)
        return result

    cls.__call__ = observed
    try:
        yield
    finally:
        cls.__call__ = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'candidate', 'input_study', 'inputs', 'feature_study', 'features',
                 'initial_study', 'initial_private', 'diagnostic_study', 'diagnostic_private',
                 'study', 'private'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite a diagnostic')
    prior = json.loads((a.diagnostic_study/'report.json').read_text())
    if prior.get('complete') is not True or prior.get('decision') != 'DIAGNOSTIC_ONLY_NO_MODEL_QUALIFICATION':
        raise ValueError('Bilingual replay diagnostic must complete first')
    prior_protocol = json.loads((a.diagnostic_study/'protocol.json').read_text())
    prior_source = a.diagnostic_study/'source.py'
    if file_sha(prior_source) != prior_protocol['source_sha256']:
        raise ValueError('Prior diagnostic source differs')
    from training.train_large_review_attention_v10 import load_inputs
    rows_by_language, bindings = load_inputs(a)
    if bindings != prior_protocol['bindings']:
        raise ValueError('Input bindings differ from prior diagnostic')
    selections = {}
    bound_files = {a.diagnostic_study/'report.json', a.diagnostic_study/'protocol.json', prior_source,
                   Path(__file__), Path('src/roguard/gemma4_attention.py')}
    for lang in rows_by_language:
        record = prior['languages'][lang]
        if record.get('complete') is not True:
            raise ValueError('Incomplete language diagnostic')
        for name, expected in record['artifacts'].items():
            path = a.diagnostic_private/name
            if Path(name).name != name or file_sha(path) != expected:
                raise ValueError('Prior diagnostic artifact differs')
            bound_files.add(path)
        observations = [json.loads(line) for line in
                        (a.diagnostic_private/(lang+'-observations.jsonl')).read_text().splitlines()]
        if len(observations) != record['checked']:
            raise ValueError('Observation count differs')
        selections[lang] = selected_cases(observations)
    file_bindings = {path: file_sha(path) for path in bound_files}
    if shutil.disk_usage(a.private.parent).free < 8*1024**3:
        raise ValueError('Less than 8 GiB free')

    import numpy as np
    import mlx.core as mx
    from safetensors.numpy import load_file
    from eval.run_trajectory_v10 import LocalFrozenV9
    from roguard.gemma4_attention import capture, capture_with_prefix, replay
    from roguard.gemma4_contract import PREFIX_TOKENS, classifier_tokens
    from roguard.gemma4_review import _Engine, _load_weights

    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    protocol = dict(format='roguard-execution-layer-diagnostic-1', bindings=bindings,
        source_sha256=file_sha(__file__), prior_report_sha256=file_bindings[a.diagnostic_study/'report.json'],
        prior_protocol_sha256=file_bindings[a.diagnostic_study/'protocol.json'],
        attention_helper_sha256=file_bindings[Path('src/roguard/gemma4_attention.py')],
        selection='First DEV control plus two largest discrepancies per path/metric; ties by feature index',
        cases={lang: [r['feature_index'] for r in selected] for lang, selected in selections.items()},
        trace='Native and replay decoder calls: final hidden token, prefix-boundary and final KV tokens; materialize after each layer',
        trace_limitation='Sparse KV probes do not bound every KV entry. Instrumentation fidelity is measured separately.',
        feature_tolerance=1e-3, instrumentation_tolerance=1e-5,
        optimizer_updates=0, held_out_test_opened=False, qualification_possible=False)
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    started = time.monotonic()
    report = dict(complete=False, languages={}, optimizer_updates=0, qualified=False)

    def persist():
        report['seconds'] = time.monotonic()-started
        tmp = a.study/'report.tmp'
        tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(a.study/'report.json')

    def error(left, right):
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('Invalid comparison arrays')
        return float(np.max(np.abs(left-right)))

    for lang, selected in selections.items():
        engine = _Engine(*frozen.models[lang]._args); model = engine.model
        _load_weights(model, load_file(a.initial_private/(lang+'-selected.safetensors')))
        model.eval()
        prefix = engine.make_cache(model.base)
        mx.eval(model.base.model(mx.array([PREFIX_TOKENS]), cache=prefix))
        references = np.load(a.features/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        records = []; report['languages'][lang] = records
        for selection in selected:
            index = selection['feature_index']; row = rows_by_language[lang][index]
            if row['case_id'] != selection['case_id'] or row['partition'] != 'dev':
                raise ValueError('Selected case identity differs')
            if classifier_tokens(engine.tokenizer, row['payload']) != row['tokens']:
                raise ValueError('Runtime tokenizer differs')
            whole = mx.array([row['tokens']]); suffix = whole[:, len(PREFIX_TOKENS):]
            values = {}; traces = {}; capture_boundary_errors = {}

            def native(cached):
                return model.base.model(suffix if cached else whole,
                    cache=deepcopy(prefix) if cached else None)[:, -1, :]

            for name, cached in (('full', False), ('prefix', True)):
                value = native(cached); mx.eval(value)
                values[name] = np.asarray(value).copy()
                state = (capture_with_prefix(model.base, suffix, prefix, start=24) if cached
                         else capture(model.base, whole, start=24))
                mx.eval(state)
                value = replay(model.base, state)[:, 0, :]; mx.eval(value)
                values[name+'_replay'] = np.asarray(value).copy()
                del value; mx.clear_cache()
                trace = {}

                def observe(layer_index, result):
                    hidden, kv, _ = result
                    probes = {'hidden': hidden[:, -1, :]}
                    for kind, tensor in zip(('key', 'value'), kv):
                        probes[kind+'_last'] = tensor[:, :, -1, :]
                        probes[kind+'_prefix_last'] = tensor[:, :, len(PREFIX_TOKENS)-1, :]
                    mx.eval(probes)
                    for key, tensor in probes.items():
                        trace[str(layer_index)+'_'+key] = np.asarray(tensor).copy()

                with observe_layers(model.base.model.layers, observe):
                    value = native(cached); mx.eval(value)
                values[name+'_traced'] = np.asarray(value).copy(); traces[name] = trace
                capture_boundary_errors[name] = error(np.asarray(state['hidden'])[:, 0, :],
                                                       trace['23_hidden'])
                trace = {}
                with observe_layers(model.base.model.layers, observe):
                    value = replay(model.base, state)[:, 0, :]; mx.eval(value)
                values[name+'_replay_traced'] = np.asarray(value).copy()
                traces[name+'_replay'] = trace
                if {int(key.split('_')[0]) for key in trace} != set(range(24, len(model.base.model.layers))):
                    raise ValueError('Replay trace omits a shared tail layer')
                del state, value; mx.clear_cache()
            if traces['full'].keys() != traces['prefix'].keys():
                raise ValueError('Native traces differ in structure')
            layer_errors = {key: error(traces['full'][key], traces['prefix'][key])
                            for key in traces['full']}
            artifact = a.private/(lang+'-'+str(index)+'-probes.npz')
            np.savez_compressed(artifact, **{name+'_'+key: value for name, trace in traces.items()
                                            for key, value in trace.items()}, **values)
            record = dict(feature_index=index, case_id=row['case_id'], tokens=len(row['tokens']),
                prior_observation=selection, artifact_sha256=file_sha(artifact),
                native_cached_vs_uncached=error(values['prefix'], values['full']),
                replay_errors={name: error(values[name], values[name+'_replay']) for name in ('prefix', 'full')},
                capture_boundary_errors=capture_boundary_errors,
                reference_errors={name: error(value, references[index][None, :]) for name, value in values.items()},
                instrumentation_errors={name: error(values[name], values[name+'_traced'])
                                        for name in ('prefix', 'full', 'prefix_replay', 'full_replay')},
                native_replay_layer_errors={name: {key: error(traces[name][key], value)
                    for key, value in traces[name+'_replay'].items()} for name in ('prefix', 'full')},
                layer_errors=layer_errors)
            records.append(record); persist()
            print(json.dumps(dict(language=lang, feature_index=index,
                native_cached_vs_uncached=record['native_cached_vs_uncached'],
                replay_errors=record['replay_errors'], instrumentation_errors=record['instrumentation_errors'])), flush=True)
            del values, traces, trace, whole, suffix; mx.clear_cache()
        del model, engine, prefix, references
        gc.collect(); mx.clear_cache()
    _, final_bindings = load_inputs(a)
    if final_bindings != bindings or any(file_sha(path) != sha for path, sha in file_bindings.items()):
        raise ValueError('Diagnostic inputs or source changed')
    report.update(complete=True, decision='DIAGNOSTIC_ONLY_NO_MODEL_QUALIFICATION')
    persist()


if __name__ == '__main__':
    main()
