"""Bind a stable release wheel to qualified code, model files and evaluation."""
import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from roguard.model_install import manifest
from roguard.gemma4_contract import FILES
from scripts.prepare_gemma4_model_download import add_terms, bundle_payload
from training.evaluate_grounding_tail_reserved_v9 import require_installed_gate
from training.shared_tail_qualification_v9 import candidate_bindings
from training.stage_grounding_tail_bundle_v9 import qualification
from training.verify_candidate_pipeline_v9 import counts
from training.verify_grounding_tail_installed_v9 import digest


def verify_wheels(qualified, release, download_raw):
    download_sha = hashlib.sha256(download_raw).hexdigest()
    spec = manifest(download_raw, download_sha)
    filename = 'model-download-'+spec['id']+'.json'
    entry = dict(manifest=filename, sha256=download_sha, runtime='apple-silicon-python312')
    catalog = dict(profiles={spec['id']: entry, 'v9': entry})
    expected_additions = {'roguard/'+filename: download_raw,
        'roguard/model-profiles.json': catalog}
    with zipfile.ZipFile(qualified) as old, zipfile.ZipFile(release) as new:
        for archive in (old, new):
            names = archive.namelist()
            if len(names) != len(set(names)) or any('..' in PurePosixPath(n).parts or '\\' in n or n.startswith('/') for n in names):
                raise ValueError('Invalid or duplicate wheel member')
        previous = {n: old.read(n) for n in old.namelist() if n.startswith('roguard/') and not n.endswith('/')}
        current = {n: new.read(n) for n in new.namelist() if n.startswith('roguard/') and not n.endswith('/')}
        if not previous or any(current.get(n) != raw for n, raw in previous.items()):
            raise ValueError('Qualified package code or data changed')
        if set(current)-set(previous) != set(expected_additions):
            raise ValueError('Only the qualified model manifest and profile catalog may be added')
        if current['roguard/'+filename] != download_raw or json.loads(current['roguard/model-profiles.json']) != catalog:
            raise ValueError('Published model catalog or manifest differs')
        metadata_names = [n for n in new.namelist() if n.endswith('.dist-info/METADATA')]
        old_metadata_names = [n for n in old.namelist() if n.endswith('.dist-info/METADATA')]
        if len(metadata_names) != 1 or len(old_metadata_names) != 1:
            raise ValueError('Expected one distribution metadata directory')
        directory = metadata_names[0].rsplit('/', 1)[0]
        old_directory = old_metadata_names[0].rsplit('/', 1)[0]
        if any(not n.startswith(('roguard/', directory+'/')) for n in new.namelist()):
            raise ValueError('Unexpected release wheel installation surface')
        metadata = BytesParser().parsebytes(new.read(metadata_names[0]))
        old_metadata = BytesParser().parsebytes(old.read(old_metadata_names[0]))
        if metadata['Name'] != 'roguard-child-safety' or metadata['Version'] != '9.0.0' or metadata['Requires-Python'] != '>=3.9':
            raise ValueError('Stable distribution identity differs')
        if sorted(metadata.get_all('Requires-Dist') or []) != sorted(old_metadata.get_all('Requires-Dist') or []):
            raise ValueError('Qualified dependency requirements changed')
        if new.read(directory+'/entry_points.txt') != old.read(old_directory+'/entry_points.txt'):
            raise ValueError('Qualified command entry points changed')
    return dict(qualified_wheel_sha256=digest(qualified), release_wheel_sha256=digest(release),
        unchanged_package_files=len(previous), added_model_files=sorted(expected_additions),
        manifest_sha256=download_sha, bundle_sha256=spec['bundle']['sha256'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualified-wheel', type=Path, required=True)
    parser.add_argument('--release-wheel', type=Path, required=True)
    parser.add_argument('--download-manifest', type=Path, required=True)
    parser.add_argument('--qualified-bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--installed-study', type=Path, required=True)
    parser.add_argument('--reserved-study', type=Path, required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    installed = args.installed_study
    reserved = args.reserved_study
    installed_report = json.loads((installed/'actual/report.json').read_text())
    result = verify_wheels(args.qualified_wheel, args.release_wheel, args.download_manifest.read_bytes())
    payload = bundle_payload(args.qualified_bundle, result['bundle_sha256'])
    terms = add_terms(payload, Path('models/v9/licenses'))
    spec = manifest(args.download_manifest.read_bytes(), result['manifest_sha256'])
    expected_assets = {name: dict(bytes=size, sha256=identity) for name, (_, size, identity) in payload.files.items()}
    expected_assets.update({'base/'+name: record for name, record in FILES.items()})
    if ({r['path']: dict(bytes=r['bytes'], sha256=r['sha256']) for r in spec['files']} != expected_assets
            or spec['terms'] != terms):
        raise ValueError('Download inventory differs from the qualified bundle and attribution')
    require_installed_gate(installed_report, result['qualified_wheel_sha256'], result['bundle_sha256'])
    _, bindings = candidate_bindings(args.candidate)
    bindings.update(qualification(root, args.prior))
    for study in (installed, reserved):
        for name, expected in json.loads((study/'source-freeze.json').read_text()).items():
            if digest(name) != expected:
                raise ValueError('Qualification source differs: '+name)
    selection = json.loads((reserved/'selection.json').read_text())
    report = json.loads((reserved/'actual/report.json').read_text())
    if (not report.get('complete') or not report.get('counts_pass')
            or report.get('selection_sha256') != digest(reserved/'selection.json')
            or selection['wheel_sha256'] != result['qualified_wheel_sha256']
            or selection['bundle_sha256'] != result['bundle_sha256']
            or selection['installed_report_sha256'] != digest(installed/'actual/report.json')):
        raise ValueError('Reserved evaluation is incomplete or bound to different assets')
    measured = {}
    for language in ('ro', 'uk'):
        values = [json.loads(line) for line in (reserved/'actual'/f'{language}-test-predictions.jsonl').read_text().splitlines()]
        original_path = root/'paired-scope-117/data'/f'{language}-test.jsonl'
        if digest(original_path) != selection['input_sha256'][language]:
            raise ValueError('Reserved input identity changed')
        originals = {v['case_id']: v for v in map(json.loads, original_path.read_text().splitlines())}
        if (set(originals) != {v['case_id'] for v in values}
                or any(v['gold'] != originals[v['case_id']]['review'] or v['language'] != language for v in values)):
            raise ValueError('Reserved prediction identities or labels changed')
        measured[language] = counts(values)
        if (len(values) != 300 or len({v['case_id'] for v in values}) != 300
                or measured[language]['positive'] != 150 or measured[language]['negative'] != 150
                or not measured[language]['passes'] or measured[language] != report['languages'][language]['counts']):
            raise ValueError('Reserved predictions fail or differ from their report')
    result.update(complete=True, development_and_prior_qualified=True, installed_execution_verified=True,
        reserved_counts=measured, synthetic_only=True, establishes_real_world_accuracy=False,
        publication_performed=False, evaluation_sha256=digest(reserved/'actual/report.json'),
        installed_report_sha256=digest(installed/'actual/report.json'), qualification_parents=bindings)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
