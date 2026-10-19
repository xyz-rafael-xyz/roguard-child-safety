"""Stage pinned Gemma4 download assets without uploading or qualifying a release.

The unchanged 5 GB base is fetched from an immutable upstream revision. Only
RoGuard adapters, manifests, and attribution files become GitHub release assets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from roguard.gemma4_contract import BASE_MODEL, BASE_REVISION, FILES, VERSIONS
from roguard.gemma4_review import Gemma4ReviewModel
from roguard.language_bundle import LanguageReviewModel, _read_manifest
from roguard.model_install import manifest, _matches
from scripts.prepare_portable_model_download import Payload, digest, encoded


def bundle_payload(bundle, expected_sha256):
    bundle = Path(bundle).resolve()
    dispatch = LanguageReviewModel(bundle, expected_sha256=expected_sha256)
    spec = _read_manifest(bundle, expected_sha256)
    expected_names = {language: dict(path=language+'-bundle.json',
        sha256=spec['languages'][language]['sha256'], device='mps') for language in ('ro', 'uk')}
    if spec['languages'] != expected_names:
        raise ValueError('Expected staged bilingual Gemma4 bundle layout')
    del dispatch
    payload = Payload()
    payload.add('bundle.json', bundle, expected_sha256)
    for language, entry in spec['languages'].items():
        path = bundle.parent/entry['path']
        model = Gemma4ReviewModel.from_bundle(path, expected_sha256=entry['sha256'])
        child = _read_manifest(path, entry['sha256'])
        if (model.language != language or child['assets'] != dict(artifact=language+'-review.json',
                adapter=language+'-adapter.safetensors', base='base')):
            raise ValueError('Expected staged Gemma4 child layout and language')
        payload.add(entry['path'], path, entry['sha256'])
        payload.add(child['assets']['artifact'], bundle.parent/child['assets']['artifact'], child['artifact_sha256'])
        payload.add(child['assets']['adapter'], bundle.parent/child['assets']['adapter'], model._spec['adapter_sha256'])
    for name, record in FILES.items():
        if not _matches(bundle.parent/'base'/name, record):
            raise ValueError('Pinned base differs before download staging: '+name)
    payload.add('runtime-constraints.txt', ''.join(name+'=='+version+'\n' for name, version in sorted(VERSIONS.items())).encode())
    return payload


def add_terms(payload, root):
    root = Path(root)
    records = json.loads((root/'gemma4-sources.json').read_text())
    wanted = {'gemma4-google-README.md', 'gemma4-mlx-README.md', 'Apache-2.0.txt'}
    records += [r for r in json.loads((root/'sources.json').read_text()) if r['file'] == 'Apache-2.0.txt']
    if len(records) != len(wanted) or {r['file'] for r in records} != wanted:
        raise ValueError('Missing pinned Gemma4 attribution')
    for record in records:
        payload.add('licenses/'+record['file'], root/record['file'], record['sha256'])
    payload.json('licenses/sources.json', records)
    payload.add('NOTICE.txt', (
        'RoGuard optional Gemma4 review model.\n'
        'Base: Google Gemma 4 E4B; MLX conversion: mlx-community.\n'
        'RoGuard modifications are separate trained attention adapters and classification heads.\n'
        'The upstream base files are unmodified and fetched from a pinned revision.\n'
        'Google identifies Gemma 4 as Apache-2.0. The conversion card retains a\n'
        'license: gemma metadata tag; both original cards are preserved in licenses/.\n'
        'Model terms: https://ai.google.dev/gemma/docs/gemma_4_license\n'
        'This staging operation does not certify accuracy or authorize external actions.\n'
    ).encode())
    return ['https://ai.google.dev/gemma/docs/gemma_4_license'] + [r['url'] for r in records]


def stage(payload, output, profile_id, release_url, terms):
    if not re.fullmatch(r'https://github\.com/xyz-rafael-xyz/roguard-child-safety/releases/download/[A-Za-z0-9._-]+', release_url):
        raise ValueError('Expected this repository\'s explicit release download URL')
    output = Path(output)
    if output.exists():
        raise ValueError('Output already exists; use a fresh staging directory')
    if 'bundle.json' not in payload.files or any(name == 'base' or name.startswith('base/') for name in payload.files):
        raise ValueError('Base files must come only from the pinned upstream revision')
    entries = [dict(path=name, url=release_url+'/'+identity+'.asset', sha256=identity, bytes=size)
        for name, (_, size, identity) in sorted(payload.files.items())]
    entries += [dict(path='base/'+name, url='https://huggingface.co/'+BASE_MODEL+'/resolve/'+BASE_REVISION+'/'+name,
        sha256=record['sha256'], bytes=record['bytes']) for name, record in sorted(FILES.items())]
    spec = dict(format='roguard-model-download-v1', id=profile_id,
        bundle=dict(path='bundle.json', sha256=payload.files['bundle.json'][2]), files=entries, terms=terms)
    raw = encoded(spec)
    identity = hashlib.sha256(raw).hexdigest()
    manifest(raw, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='roguard-gemma4-assets-', dir=output.parent) as temporary:
        root = Path(temporary)
        assets = root/'assets'
        assets.mkdir()
        for name, (source, size, expected) in payload.files.items():
            target = root/'payload'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(source, bytes):
                target.write_bytes(source)
            else:
                shutil.copyfile(source, target)
            if target.stat().st_size != size or digest(target) != expected:
                raise ValueError('Payload changed during staging: '+name)
            asset = assets/(expected+'.asset')
            if not asset.exists():
                os.link(target, asset)
        filename = 'model-download-'+profile_id+'.json'
        (root/filename).write_bytes(raw)
        (root/'model-profiles.json').write_bytes(encoded(dict(profiles={profile_id: dict(
            manifest=filename, sha256=identity, runtime='apple-silicon-python312')})))
        (root/'staging-report.json').write_bytes(encoded(dict(manifest_sha256=identity,
            bundle_sha256=spec['bundle']['sha256'], files=len(entries),
            upstream_bytes=sum(r['bytes'] for r in FILES.values()),
            logical_bytes=sum(r['bytes'] for r in entries),
            upload_bytes=sum(p.stat().st_size for p in assets.iterdir()),
            uploaded=False, qualifies_for_release=False)))
        root.rename(output)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--licenses', type=Path, default=Path('models/v9/licenses'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--release-url', required=True)
    args = parser.parse_args()
    payload = bundle_payload(args.bundle, args.bundle_sha256)
    terms = add_terms(payload, args.licenses)
    spec = stage(payload, args.output, args.id, args.release_url, terms)
    print(json.dumps(dict(files=len(spec['files']), output=str(args.output), uploaded=False)))


if __name__ == '__main__':
    main()
