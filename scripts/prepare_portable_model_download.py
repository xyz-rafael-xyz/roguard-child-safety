"""Stage pinned portable model assets for a later, separately qualified release.

This command never uploads files, edits the package catalog, or declares a
candidate qualified. It accepts completed experiments for private installation
checks as well as qualified candidates. Qualification evidence is handled by the
study verifiers before publication.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from roguard.entailment_bundle import load_bundle
from roguard.model_install import manifest, _relative
from roguard.native_review import BASES, WEIGHT_FILES, NativeReviewModel
from roguard.nli_v21 import BASE_HASHES
from roguard.runtime_requirements import runtime_extra


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=True) + '\n').encode()


class Payload:
    def __init__(self):
        self.files = {}

    def add(self, name, source, expected=None):
        _relative(name)
        if name in self.files:
            raise ValueError('Duplicate payload path: ' + name)
        if isinstance(source, bytes):
            size, identity = len(source), hashlib.sha256(source).hexdigest()
        else:
            source = Path(source)
            size, identity = source.stat().st_size, digest(source)
        if not 0 < size < 2_000_000_000:
            raise ValueError('Payload exceeds individual release asset limit')
        if expected is not None and identity != expected:
            raise ValueError('Source identity differs: ' + name)
        self.files[name] = (source, size, identity)
        return identity

    def json(self, name, value):
        return self.add(name, encoded(value))


def completed(candidate, artifact):
    record = json.loads((candidate / 'development.json').read_text())
    identity = digest(candidate / artifact)
    if record.get('complete') is not True or record.get('artifact_sha256') != identity:
        raise ValueError('Candidate must be complete and bound to its selected artifact')
    return identity


def native_payload(candidates, bases, base_manifest):
    result = Payload()
    languages = {}
    inventory = json.loads(base_manifest.read_text())
    for language in ('ro', 'uk'):
        candidate, base = candidates[language], bases[language]
        identity = completed(candidate, 'review.json.gz')
        spec = json.loads(gzip.decompress((candidate / 'review.json.gz').read_bytes()))
        # Lazy construction checks the pinned artifact and tokenizer contracts.
        NativeReviewModel(candidate / 'review.json.gz', candidate / 'selected-model.safetensors',
                          base, base_manifest, expected_artifact_sha256=identity)
        if spec['language'] != language:
            raise ValueError('Candidate language differs')
        result.add(language + '/review.json.gz', candidate / 'review.json.gz', identity)
        result.add(language + '/model.safetensors', candidate / 'selected-model.safetensors', spec['checkpoint_sha256'])
        result.add(language + '/base-manifest.json', base_manifest, spec['base_manifest_sha256'])
        for name in sorted(BASES[language][3] - WEIGHT_FILES):
            result.add(language + '/base/' + name, base / name,
                       inventory['models'][language]['files'][name]['sha256'])
        child = language + '/bundle.json'
        child_sha = result.json(child, dict(format='roguard-native-review-bundle-v1',
            artifact_sha256=identity, assets=dict(artifact='review.json.gz', checkpoint='model.safetensors',
                                                 base='base', manifest='base-manifest.json')))
        languages[language] = dict(path=child, sha256=child_sha, device='cpu')
    result.json('bundle.json', dict(format='roguard-language-review-bundle-v1', languages=languages))
    return result


def entailment_payload(candidate, base):
    result = Payload()
    identity = completed(candidate, 'review.json')
    spec = json.loads((candidate / 'review.json').read_text())
    kinds = {'roguard-paired-nli-artifact-v1': 'review',
             'roguard-policy-entailment-artifact-v1': 'support-revision',
             'roguard-reason-entailment-artifact-v1': 'separate-reasons'}
    if spec.get('format') not in kinds:
        raise ValueError('Unsupported entailment candidate')
    result.add('review.json', candidate / 'review.json', identity)
    result.add('model.safetensors', candidate / 'selected-model.safetensors', spec['checkpoint_sha256'])
    for name, expected in sorted(BASE_HASHES.items()):
        if name != 'model.safetensors':
            result.add('base/' + name, base / name, expected)
    result.json('bundle.json', dict(format='roguard-entailment-review-bundle-v1', kind=kinds[spec['format']],
        artifact=dict(path='review.json', sha256=identity), checkpoint='model.safetensors', base='base'))
    return result


def add_terms(payload, license_root, kind):
    wanted = ({'ro-UPSTREAM-LICENSE', 'Apache-2.0.txt', 'ro-README.md', 'uk-README.md'}
              if kind == 'native' else {'nli-UPSTREAM-LICENSE.txt'})
    records = [r for r in json.loads((license_root / 'sources.json').read_text()) if r['file'] in wanted]
    if {r['file'] for r in records} != wanted:
        raise ValueError('Missing upstream license provenance')
    for record in records:
        payload.add('licenses/' + record['file'], license_root / record['file'], record['sha256'])
    payload.json('licenses/sources.json', records)
    payload.add('NOTICE.txt', (('RoGuard optional ' + kind + ' model files.\n'
        'The upstream terms and provenance are in licenses/.\n'
        'Packaging does not certify model accuracy or authorize external actions.\n').encode()))
    return [r['url'] for r in records]


def stage(payload, output, profile_id, release_url, terms, *, validate_bundle=True):
    if not re.fullmatch(r'https://github\.com/xyz-rafael-xyz/roguard-child-safety/releases/download/[A-Za-z0-9._-]+', release_url):
        raise ValueError('Expected this repository\'s explicit GitHub release download URL')
    output = Path(output)
    if output.exists():
        raise ValueError('Output already exists; use a fresh staging directory')
    entries = [dict(path=name, url=release_url + '/' + identity + '.asset', sha256=identity, bytes=size)
               for name, (_, size, identity) in sorted(payload.files.items())]
    spec = dict(format='roguard-model-download-v1', id=profile_id,
                bundle=dict(path='bundle.json', sha256=payload.files['bundle.json'][2]), files=entries, terms=terms)
    raw = encoded(spec)
    sha = hashlib.sha256(raw).hexdigest()
    manifest(raw, sha)  # Reject path collisions and invalid sizes before writes.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='roguard-assets-', dir=output.parent) as temporary:
        root = Path(temporary)
        local = root / 'payload'
        assets = root / 'assets'
        assets.mkdir()
        for name, (source, size, identity) in payload.files.items():
            target = local / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(source, bytes):
                target.write_bytes(source)
            else:
                shutil.copyfile(source, target)
            if target.stat().st_size != size or digest(target) != identity:
                raise ValueError('Payload changed during staging: ' + name)
            asset = assets / (identity + '.asset')
            if not asset.exists():
                os.link(target, asset)
        if validate_bundle:
            if runtime_extra(local / 'bundle.json', spec['bundle']['sha256']) != 'semantic':
                raise ValueError('Expected a portable semantic runtime')
            model = load_bundle(local / 'bundle.json', expected_sha256=spec['bundle']['sha256'])
            # Language dispatch is lazy; explicitly validate both children too.
            bundle = json.loads((local / 'bundle.json').read_text())
            for entry in bundle.get('languages', {}).values():
                load_bundle(local / entry['path'], expected_sha256=entry['sha256'])
            del model
        (root / 'model-download.json').write_bytes(raw)
        (root / 'model-profiles.json').write_bytes(encoded(dict(profiles={profile_id: dict(
            manifest='model-download.json', sha256=sha, runtime='portable')})))
        (root / 'staging-report.json').write_bytes(encoded(dict(manifest_sha256=sha,
            logical_bytes=sum(e['bytes'] for e in entries), upload_bytes=sum(p.stat().st_size for p in assets.iterdir()),
            files=len(entries), assets=len(list(assets.iterdir())), uploaded=False, qualifies_for_release=False)))
        root.rename(output)
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('native', 'entailment'), required=True)
    p.add_argument('--candidate', type=Path, required=True, help='Bilingual candidate directory, or parent of native ro/uk directories')
    p.add_argument('--base', type=Path)
    p.add_argument('--ro-base', type=Path)
    p.add_argument('--uk-base', type=Path)
    p.add_argument('--base-manifest', type=Path, default=Path('research/v9-stabilization/native-full-22-base-manifest.json'))
    p.add_argument('--licenses', type=Path, default=Path('models/v9/licenses'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--release-url', required=True)
    a = p.parse_args()
    if a.kind == 'native':
        if not a.ro_base or not a.uk_base:
            p.error('Native staging requires --ro-base and --uk-base')
        payload = native_payload({l: a.candidate / l for l in ('ro', 'uk')},
                                 dict(ro=a.ro_base, uk=a.uk_base), a.base_manifest)
    else:
        if not a.base:
            p.error('Entailment staging requires --base')
        payload = entailment_payload(a.candidate, a.base)
    terms = add_terms(payload, a.licenses, a.kind)
    spec = stage(payload, a.output, a.id, a.release_url, terms)
    print(json.dumps(dict(files=len(spec['files']), output=str(a.output), uploaded=False)))


if __name__ == '__main__':
    main()
