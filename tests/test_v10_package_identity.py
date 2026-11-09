import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest
from scripts.verify_v10_package_identity import verify_wheels

D = 'roguard_child_safety-10.0.0.dist-info/'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_wheel(path, files, *, corrupt_record=False):
    files = dict(files)
    rows = []
    for name, raw in files.items():
        hashed = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        rows.append([name, 'sha256=' + hashed, str(len(raw))])
    rows.append([D + 'RECORD', '', ''])
    if corrupt_record:
        rows[0][1] = 'sha256=' + 'X' * 43
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    files[D + 'RECORD'] = output.getvalue().encode()
    with zipfile.ZipFile(path, 'w') as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return sha(path.read_bytes())


@pytest.fixture
def pair(tmp_path):
    bundle = b'{"synthetic":true}'
    spec = {'format': 'roguard-model-download-v1', 'id': 'v10-synthetic',
            'bundle': {'path': 'bundle.json', 'sha256': sha(bundle)},
            'files': [{'path': 'bundle.json', 'sha256': sha(bundle), 'bytes': len(bundle),
                       'url': 'https://example.invalid/bundle.json'}],
            'terms': ['https://example.invalid/terms']}
    raw = json.dumps(spec).encode()
    old_catalog = {'profiles': {'v9': {'manifest': 'old.json', 'sha256': '0' * 64,
                                    'runtime': 'apple-silicon-python312'}}}
    old = {'roguard/__init__.py': b'# frozen code\n',
           'roguard/model-profiles.json': json.dumps(old_catalog).encode(),
           D + 'METADATA': b'Metadata-Version: 2.4\nName: roguard-child-safety\nVersion: 10.0.0\nRequires-Python: >=3.9\n',
           D + 'entry_points.txt': b'[console_scripts]\nroguard-v10=roguard.trajectory_cli:main\n',
           D + 'WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
           D + 'licenses/LICENSE': b'synthetic license\n'}
    current = dict(old)
    entry = {'manifest': 'model-download-v10-synthetic.json', 'sha256': sha(raw),
             'runtime': 'apple-silicon-python312'}
    new_catalog = {'profiles': dict(old_catalog['profiles'], **{'v10': entry, 'v10-synthetic': entry})}
    current['roguard/model-profiles.json'] = json.dumps(new_catalog).encode()
    current['roguard/model-download-v10-synthetic.json'] = raw
    return tmp_path, old, current, raw


def run(pair, *, corrupt_record=False):
    root, old, current, manifest = pair
    a = root / 'qualified.whl'
    b = root / 'release.whl'
    return verify_wheels(a, b, manifest,
                         expected_qualified_sha256=write_wheel(a, old),
                         expected_release_sha256=write_wheel(b, current, corrupt_record=corrupt_record),
                         expected_manifest_sha256=sha(manifest))


def test_exact_code_with_additive_catalog_passes_without_qualification(pair):
    result = run(pair)
    assert result['package_identity_pass'] is True
    assert result['model_qualification_verified'] is False
    assert result['publication_performed'] is False


@pytest.mark.parametrize('name', ['roguard/__init__.py', D + 'METADATA', D + 'entry_points.txt',
                                D + 'WHEEL', D + 'licenses/LICENSE'])
def test_changed_qualified_surface_rejected(pair, name):
    pair[2][name] += b'changed\n'
    with pytest.raises(ValueError, match='changed|differs'):
        run(pair)


def test_preserve_v9_catalog_entry(pair):
    catalog = json.loads(pair[2]['roguard/model-profiles.json'])
    catalog['profiles']['v9']['sha256'] = '1' * 64
    pair[2]['roguard/model-profiles.json'] = json.dumps(catalog).encode()
    with pytest.raises(ValueError, match='Catalog changed'):
        run(pair)


def test_added_python_module_rejected(pair):
    pair[2]['roguard/new_loader.py'] = b'changed code\n'
    with pytest.raises(ValueError, match='added or removed'):
        run(pair)


def test_record_must_match_content(pair):
    with pytest.raises(ValueError, match='RECORD content'):
        run(pair, corrupt_record=True)


@pytest.mark.parametrize('name', ['roguard/../escape.py', 'roguard/X.py', 'roguard/__init__.py/child'])
def test_unsafe_paths_and_case_aliases_rejected(pair, name):
    if name == 'roguard/X.py':
        pair[2]['roguard/x.py'] = b'a'
    pair[2][name] = b'b'
    with pytest.raises(ValueError, match='Unsafe|Case-aliased|collision'):
        run(pair)


def test_final_version_must_precede_qualification(pair):
    for files in pair[1:3]:
        files[D + 'METADATA'] = files[D + 'METADATA'].replace(b'Version: 10.0.0', b'Version: 9.0.0')
    with pytest.raises(ValueError, match='Final distribution identity'):
        run(pair)


def test_unicode_normalization_alias_rejected(pair):
    for files in pair[1:3]:
        files['roguard/caf\u00e9.py'] = b'one'
        files['roguard/cafe\u0301.py'] = b'two'
    with pytest.raises(ValueError, match='Unsafe'):
        run(pair)


def test_wrong_frozen_wheel_digest_rejected(pair):
    root, old, current, raw = pair
    a = root / 'a.whl'
    b = root / 'b.whl'
    write_wheel(a, old)
    identity = write_wheel(b, current)
    with pytest.raises(ValueError, match='Wheel identity'):
        verify_wheels(a, b, raw, expected_qualified_sha256='f' * 64,
                      expected_release_sha256=identity, expected_manifest_sha256=sha(raw))
