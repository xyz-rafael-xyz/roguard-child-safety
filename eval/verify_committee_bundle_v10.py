"""Check a relocated installed committee bundle against fresh study-29 outputs."""
import argparse
import gc
import json
from pathlib import Path
import time
import zipfile

from roguard.committee_bundle import load_committee
from roguard.gemma4_committee import COMPONENTS
from roguard.trajectory_data import file_sha, prefix, read_jsonl, write_jsonl
from roguard.trajectory_evaluation import checkpoint_id
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'base', 'selector', 'runtime_study', 'installed_wheel', 'study'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--sha256', required=True)
    a = p.parse_args()
    if a.study.exists():
        raise ValueError('Refusing to overwrite installed bundle verification')
    import roguard.committee_bundle as module
    installed = Path(module.__file__).resolve()
    if installed == Path('src/roguard/committee_bundle.py').resolve():
        raise ValueError('Expected installed bundle loader, not source checkout')
    checked = 0
    with zipfile.ZipFile(a.installed_wheel) as wheel:
        for name in wheel.namelist():
            if name.startswith('roguard/') and not name.endswith('/'):
                if (installed.parent.parent/name).read_bytes() != wheel.read(name):
                    raise ValueError('Installed package bytes differ from wheel')
                checked += 1
    previous = json.loads((a.runtime_study/'report.json').read_text())
    protocol = json.loads((a.runtime_study/'protocol.json').read_text())
    if not previous['complete'] or previous['decision'] != 'BINARY_RUNTIME_ENGINEERING_PASS_ONLY':
        raise ValueError('Expected completed passing runtime verification')
    selector = TrajectoryModel(a.selector)
    if selector.identity != protocol['selector_sha256']:
        raise ValueError('Memory selector differs')
    root = Path('research/v10-evidence')
    paths = {'broad': root/'broad-v9-regression/cases.jsonl',
             **{n: root/'data'/(n+'.jsonl') for n in ('dev', 'challenge', 'v9-regression')}}
    selected = []; records = {}
    for population, path in paths.items():
        if file_sha(path) != protocol['inputs'][population]:
            raise ValueError('Input population differs')
        predictions = a.runtime_study/(population+'-predictions.jsonl')
        if file_sha(predictions) != previous['prediction_sha256'][population]:
            raise ValueError('Prior runtime predictions differ')
        rows = read_jsonl(predictions)
        lookup = {checkpoint_id(c, point): (c, point) for c in read_jsonl(path) for point in c['checkpoints']}
        patterns = set()
        for row in rows:
            key = (row['language'], tuple(row['runtime']['components'][k]['predicted'] for k in COMPONENTS))
            # First occurrence of every vote pattern in each population/language,
            # plus every fixed v9 error. This is runtime parity, not a new test.
            if key not in patterns or row['baseline_predicted'] != row['gold']:
                patterns.add(key); selected.append((population, row)); records[(population, row['checkpoint_id'])] = lookup[row['checkpoint_id']]
    a.study.mkdir(parents=True); results = []; started = time.monotonic()
    for lang in ('ro', 'uk'):
        model = load_committee(a.bundle, expected_sha256=a.sha256, base=a.base, language=lang)
        if model.identity != protocol['committee_identities'][lang]:
            raise ValueError('Bundle committee identity differs')
        for population, expected in (r for r in selected if r[1]['language'] == lang):
            case, point = records[(population, expected['checkpoint_id'])]
            view, inference = prepare_memory(prefix(case, point['focus']), selector, 1200, preserve_complete=True)
            if view['mandatory_overflow']:
                raise ValueError('Previously supported prefix now overflows')
            actual = model.predict(inference)
            if actual != expected['runtime']:
                raise ValueError('Relocated installed bundle output differs from study 29')
            results.append({'population': population, 'checkpoint_id': expected['checkpoint_id'],
                            'language': lang, 'all_output_fields_exact': True, 'runtime': actual})
        del model
        import mlx.core as mx
        gc.collect(); mx.clear_cache()
    write_jsonl(a.study/'predictions.jsonl', results)
    report = {'complete': True, 'source_sha256': file_sha(Path(__file__)),
        'bundle_manifest_sha256': a.sha256, 'installed_wheel_sha256': file_sha(a.installed_wheel),
        'installed_package_files_verified': checked, 'runtime_report_sha256': file_sha(a.runtime_study/'report.json'),
        'fresh_bundle_inferences': len(results), 'all_output_fields_exact': True,
        'prediction_sha256': file_sha(a.study/'predictions.jsonl'), 'seconds': time.monotonic()-started,
        'sample_selection': 'First per population/language/vote pattern, plus every prior baseline error',
        'accuracy_test': False, 'independent_validation': False, 'stable_release_ready': False}
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
