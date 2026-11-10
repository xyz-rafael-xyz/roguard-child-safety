"""Verify release package identity only; this does not qualify a model.

The unpublished qualifying wheel must already use version 10.0.0. The only
allowed later changes are the model download manifest, additive profile catalog
entries, and their wheel RECORD checksums. Inference code and metadata stay exact.
"""
import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import unicodedata
import zipfile

from roguard.model_install import manifest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def wheel(path, expected_sha256):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 100_000_000,
            'Expected bounded code-only wheel')
    raw = path.read_bytes()
    require(sha(raw) == expected_sha256, 'Wheel identity differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(set(names)), 'Duplicate wheel member')
        aliases = {name.casefold() for name in names}
        require(len(aliases) == len(names), 'Case-aliased wheel members')
        require(sum(entry.file_size for entry in entries) <= 150_000_000,
                'Uncompressed wheel exceeds bound')
        for entry in entries:
            name = entry.filename
            require(name and unicodedata.normalize('NFC', name) == name
                    and not name.startswith('/') and '\\' not in name
                    and ':' not in name and all(p not in ('', '.', '..') for p in name.split('/')),
                    'Unsafe wheel member')
            mode = entry.external_attr >> 16
            require(not stat.S_ISLNK(mode) and not entry.is_dir(), 'Non-file wheel member')
            require(not any(str(parent).casefold() in aliases for parent in PurePosixPath(name).parents),
                    'Wheel file/directory collision')
        files = {name: archive.read(name) for name in names}
    metas = [name for name in names if name.endswith('.dist-info/METADATA')]
    require(len(metas) == 1, 'One distribution metadata directory required')
    directory = metas[0].rsplit('/', 1)[0]
    require(directory == 'roguard_child_safety-10.0.0.dist-info', 'Final version directory differs')
    require(all(name.startswith(('roguard/', directory + '/')) for name in names),
            'Unexpected wheel installation surface')
    metadata = BytesParser().parsebytes(files[metas[0]])
    for field, expected in [('Name', 'roguard-child-safety'), ('Version', '10.0.0'),
                            ('Requires-Python', '>=3.9')]:
        require(metadata.get_all(field) == [expected], 'Final distribution identity differs: ' + field)
    require('roguard/__init__.py' in files and directory + '/entry_points.txt' in files,
            'Package and command entry points required')
    record_name = directory + '/RECORD'
    require(record_name in files, 'Wheel RECORD missing')
    rows = list(csv.reader(io.StringIO(files[record_name].decode('utf-8'))))
    require(all(len(row) == 3 for row in rows), 'Malformed wheel RECORD')
    require(len(rows) == len(files) and {row[0] for row in rows} == set(files),
            'RECORD must cover every member exactly once')
    for name, digest, size in rows:
        if name == record_name:
            require(digest == size == '', 'RECORD must not hash itself')
        else:
            encoded = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).decode('ascii').rstrip('=')
            require(digest == 'sha256=' + encoded and size == str(len(files[name])),
                    'Wheel RECORD content differs: ' + name)
    return files, directory


def verify_wheels(qualified, release, download_raw, *, expected_qualified_sha256,
                  expected_release_sha256, expected_manifest_sha256):
    spec = manifest(download_raw, expected_manifest_sha256)
    require(spec['id'].startswith('v10-') and spec['id'] != 'v10',
            'A versioned v10 model profile ID is required')
    previous, directory = wheel(qualified, expected_qualified_sha256)
    current, new_directory = wheel(release, expected_release_sha256)
    require(directory == new_directory, 'Distribution identity changed')
    catalog_name = 'roguard/model-profiles.json'
    manifest_name = 'roguard/model-download-' + spec['id'] + '.json'
    record_name = directory + '/RECORD'
    require(catalog_name in previous and manifest_name not in previous,
            'Qualifying wheel must retain old catalog and omit new download manifest')
    catalog = json.loads(previous[catalog_name], object_pairs_hook=unique)
    require(type(catalog) is dict and set(catalog) == {'profiles'}
            and type(catalog['profiles']) is dict and 'v9' in catalog['profiles'],
            'Existing v9 catalog required')
    require(not {'v10', spec['id']} & set(catalog['profiles']), 'V10 profile already exists')
    entry = dict(manifest=PurePosixPath(manifest_name).name,
                 sha256=expected_manifest_sha256, runtime='apple-silicon-python312')
    expected_catalog = {'profiles': dict(catalog['profiles'], **{spec['id']: entry, 'v10': entry})}
    require(set(current) == set(previous) | {manifest_name}, 'Unexpected added or removed wheel files')
    require(current[manifest_name] == download_raw, 'Download manifest bytes differ')
    require(json.loads(current[catalog_name], object_pairs_hook=unique) == expected_catalog,
            'Catalog changed beyond the reviewed additive v10 entries')
    for name, raw in previous.items():
        if name not in (catalog_name, record_name):
            require(current[name] == raw, 'Qualified wheel member changed: ' + name)
    return dict(format='roguard-v10-package-identity-verification-1',
                package_identity_pass=True, model_qualification_verified=False,
                publication_performed=False, qualified_wheel_sha256=expected_qualified_sha256,
                release_wheel_sha256=expected_release_sha256,
                download_manifest_sha256=expected_manifest_sha256,
                bundle_sha256=spec['bundle']['sha256'],
                python_files=sum(n.startswith('roguard/') and n.endswith('.py') for n in previous),
                unchanged_files=len(previous) - 2,
                allowed_changes=[catalog_name, manifest_name, record_name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('qualified-wheel', 'release-wheel', 'download-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('qualified-wheel', 'release-wheel', 'download-manifest'):
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args()
    require(args.download_manifest.stat().st_size <= 1_000_000, 'Manifest exceeds bound')
    result = verify_wheels(args.qualified_wheel, args.release_wheel,
                          args.download_manifest.read_bytes(),
                          expected_qualified_sha256=args.qualified_wheel_sha256,
                          expected_release_sha256=args.release_wheel_sha256,
                          expected_manifest_sha256=args.download_manifest_sha256)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
