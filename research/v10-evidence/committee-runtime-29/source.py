"""Fresh end-to-end binary committee inference and component parity checks."""
import argparse
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_committee import Gemma4ReviewCommittee, COMPONENTS
from roguard.gemma4_review import Gemma4ReviewModel
from roguard.interpreter import prepare_input
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl, prefix, input_identity
from roguard.trajectory_evaluation import checkpoint_id
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'candidate', 'readout_artifacts', 'attention_artifacts', 'selector', 'study'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    p.add_argument('--installed-wheel', type=Path)
    a = p.parse_args(); root = Path('research/v10-evidence')
    if a.study.exists():
        raise ValueError('Refusing to overwrite fresh committee verification')
    previous = root/'review-unanimity-28'
    accepted = json.loads((previous/'report.json').read_text())
    if not accepted['complete'] or accepted['decision'] != 'BINARY_ENGINEERING_PASS_ONLY':
        raise ValueError('Expected the fixed passing binary-component hypothesis')
    paths = {'broad': root/'broad-v9-regression/cases.jsonl',
             **{n: root/'data'/(n+'.jsonl') for n in ('dev', 'challenge', 'v9-regression')}}
    sources = {'v9': root/'semantic-regression-11', 'readout': root/'semantic-regression-11',
               'attention': root/'semantic-tail-regression-14'}
    parent_protocols = {k: json.loads((path/'protocol.json').read_text()) for k, path in sources.items()}
    attention_report = json.loads((sources['attention']/'report.json').read_text())
    selector = TrajectoryModel(a.selector)
    for value in parent_protocols.values():
        if value['selector_sha256'] != selector.identity or any(value['inputs'][k] != file_sha(path) for k, path in paths.items()):
            raise ValueError('Parent population or memory selector differs')
    cases = {k: read_jsonl(path) for k, path in paths.items()}
    expected = {}; prediction_paths = {}
    for kind, source in sources.items():
        expected[kind] = {}
        for name in paths:
            path = source/(name+('-baseline.jsonl' if kind == 'v9' else '-candidate.jsonl'))
            if kind == 'attention' and attention_report['prediction_sha256'][name] != file_sha(path):
                raise ValueError('Bound attention predictions differ')
            rows = read_jsonl(path); expected[kind][name] = {r['checkpoint_id']: r for r in rows}
            if len(rows) != len(expected[kind][name]):
                raise ValueError('Duplicate parent prediction')
            prediction_paths[str(path)] = file_sha(path)
    a.study.mkdir(parents=True)
    import roguard.gemma4_committee as runtime_module
    runtime_path = Path(runtime_module.__file__).resolve()
    checkout_path = Path('src/roguard/gemma4_committee.py').resolve()
    if file_sha(runtime_path) != file_sha(checkout_path):
        raise ValueError('Imported committee runtime differs from the reviewed source')
    if a.installed_wheel is not None and runtime_path == checkout_path:
        raise ValueError('Installed-wheel verification imported the source checkout')
    installed_files_checked = 0
    if a.installed_wheel is not None:
        import zipfile
        with zipfile.ZipFile(a.installed_wheel) as wheel:
            for name in wheel.namelist():
                if name.startswith('roguard/') and not name.endswith('/'):
                    installed = runtime_path.parent.parent/name
                    if installed.read_bytes() != wheel.read(name):
                        raise ValueError('Installed package file differs from the bound wheel: '+name)
                    installed_files_checked += 1
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    protocol = {'format': 'roguard-committee-runtime-1', 'source_sha256': file_sha(Path(__file__)),
        'runtime_source_sha256': file_sha('src/roguard/gemma4_committee.py'),
        'amendment_sha256': file_sha(root/'protocol-amendment-27.json'),
        'binary_hypothesis_report_sha256': file_sha(previous/'report.json'),
        'inputs': {k: file_sha(v) for k, v in paths.items()}, 'parent_predictions': prediction_paths,
        'selector_sha256': selector.identity, 'memory_budget': 1200, 'preserve_complete_turns': True,
        'batch_limit': 64, 'component_specs': {}, 'fresh_inference': True,
        'fitting_performed': False, 'independent_test': False, 'stable_release_ready': False}
    if a.installed_wheel is not None:
        from importlib.metadata import version
        protocol['installed_wheel_sha256'] = file_sha(a.installed_wheel)
        protocol['installed_distribution_version'] = version('roguard-child-safety')
        protocol['runtime_loaded_from_source_checkout'] = False
        protocol['installed_package_files_verified'] = installed_files_checked
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'runtime-source.py').write_bytes(Path('src/roguard/gemma4_committee.py').read_bytes())
    outputs = {name: [] for name in paths}; maximum = {k: 0. for k in COMPONENTS}
    started = time.monotonic(); done = 0; inference_seconds = 0.
    for lang in ('ro', 'uk'):
        components = {'v9': frozen.models[lang]}
        for kind, study, private in (('readout', root/'semantic-readout-11', a.readout_artifacts),
                                      ('attention', root/'semantic-tail-14', a.attention_artifacts)):
            path = study/(lang+'-review.json'); sha = file_sha(path)
            if str(path) not in json.loads((previous/'protocol.json').read_text())['component_specs'] or sha != json.loads((previous/'protocol.json').read_text())['component_specs'][str(path)]:
                raise ValueError('Component differs from the frozen binary hypothesis')
            protocol['component_specs'][str(path)] = sha
            components[kind] = Gemma4ReviewModel(path, private/(lang+'-selected.safetensors'), a.base, expected_sha256=sha)
        model = Gemma4ReviewCommittee(**components)
        pipeline = digest({'committee': model.identity, 'selector': selector.identity,
                           'memory_budget': 1200, 'selection': 'all-prefixes-preserve-complete-turns-v1'})
        protocol.setdefault('committee_identities', {})[lang] = model.identity
        (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
        records = []
        for population, population_cases in cases.items():
            for case in (c for c in population_cases if c['language'] == lang):
                for point in case['checkpoints']:
                    cid = checkpoint_id(case, point); payload = prefix(case, point['focus'])
                    view, inference = prepare_memory(payload, selector, 1200, preserve_complete=True)
                    if view['mandatory_overflow'] or len(inference['turns']) > 12:
                        raise ValueError('Unsupported original memory view')
                    for kind in COMPONENTS:
                        parent = expected[kind][population][cid]
                        if parent['input_digest'] != input_identity(payload) or parent['inference_input_digest'] != input_identity(inference):
                            raise ValueError('Parent original or model input identity differs')
                    records.append((population, cid, payload, inference, point['review']))
        for start in range(0, len(records), 64):
            batch = records[start:start+64]; tick = time.monotonic()
            predictions = model.predict_many([r[3] for r in batch])
            inference_seconds += time.monotonic()-tick
            for record, result in zip(batch, predictions):
                population, cid, payload, inference, gold = record
                if (result['input_digest'] != prepare_input(inference)[2] or result['model_identity'] != model.identity
                        or result['component_identities'] != model.component_identities
                        or result['language'] != lang or result['focus'] != len(inference['turns'])-1):
                    raise ValueError('Fresh runtime output is bound to another input or model')
                if any(result[k] for k in ('qualified', 'establishes_safety', 'permits_external_action', 'permits_notification')):
                    raise ValueError('Experimental runtime claims qualification or external authority')
                if result['predicted'] is None or result['components'] is None:
                    raise ValueError('Fresh committee unexpectedly abstained')
                for kind in COMPONENTS:
                    parent = expected[kind][population][cid]; actual = result['components'][kind]
                    difference = abs(actual['score']-parent['score']); maximum[kind] = max(maximum[kind], difference)
                    if difference > 1e-5 or actual['predicted'] != parent['predicted']:
                        raise ValueError('Fresh committee component differs from the frozen parent: '+kind)
                reference = int(all(expected[k][population][cid]['predicted'] == 1 for k in COMPONENTS))
                if result['predicted'] != reference:
                    raise ValueError('Fresh committee vote differs from frozen decision arithmetic')
                outputs[population].append({'checkpoint_id': cid, 'language': lang, 'gold': gold,
                    'input_digest': input_identity(payload), 'inference_input_digest': input_identity(inference),
                    'pipeline_digest': pipeline, 'runtime': result,
                    'baseline_predicted': expected['v9'][population][cid]['predicted']})
                done += 1
            progress = {'done': done, 'total': 6002, 'language': lang,
                        'seconds': time.monotonic()-started, 'maximum_component_score_error': maximum}
            (a.study/'progress.json').write_text(json.dumps(progress, indent=2)+'\n'); print(json.dumps(progress), flush=True)
            for name, rows in outputs.items():
                write_jsonl(a.study/(name+'-predictions.jsonl'), rows)
        del model, components
        import gc
        import mlx.core as mx
        gc.collect(); mx.clear_cache()
    if done != 6002:
        raise ValueError('Fresh checkpoint count differs')
    populations = {}
    for name, rows in outputs.items():
        expected_ids = {checkpoint_id(c, p) for c in cases[name] for p in c['checkpoints']}
        if {r['checkpoint_id'] for r in rows} != expected_ids or len(rows) != len(expected_ids):
            raise ValueError('Incomplete fresh population')
        errors = [r['checkpoint_id'] for r in rows if r['runtime']['predicted'] != r['gold']]
        fixed = [r['checkpoint_id'] for r in rows if r['baseline_predicted'] != r['gold'] and r['runtime']['predicted'] == r['gold']]
        introduced = [r['checkpoint_id'] for r in rows if r['baseline_predicted'] == r['gold'] and r['runtime']['predicted'] != r['gold']]
        populations[name] = {'rows': len(rows), 'errors': errors, 'fixed': fixed, 'introduced': introduced}
    report = {'complete': True, 'fresh_checkpoint_inferences': done, 'maximum_component_score_error': maximum,
        'populations': populations, 'seconds': time.monotonic()-started, 'api_batch_inference_seconds': inference_seconds,
        'latency_is_per_request_benchmark': False, 'binary_component_only': True,
        'prediction_sha256': {k: file_sha(a.study/(k+'-predictions.jsonl')) for k in paths},
        'decision': 'REJECT' if any(v['errors'] for v in populations.values()) else 'BINARY_RUNTIME_ENGINEERING_PASS_ONLY',
        'independent_test': False, 'stable_release_ready': False}
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
