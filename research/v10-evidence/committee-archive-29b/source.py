"""Assemble and verify an explicitly unpublished binary-component archive."""
import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile
import zipfile

from roguard.trajectory_data import file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'wheel', 'packaging_report', 'runtime_report', 'output', 'report'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.report.exists():
        raise ValueError('Refusing to overwrite a research archive or its record')
    packaging = json.loads(a.packaging_report.read_text()); runtime = json.loads(a.runtime_report.read_text())
    if (not packaging['complete'] or not runtime['complete'] or not runtime['all_output_fields_exact']
            or runtime['bundle_manifest_sha256'] != packaging['manifest_sha256']
            or runtime['installed_wheel_sha256'] != file_sha(a.wheel)):
        raise ValueError('Expected the exact verified bundle and installed wheel')
    files = {}
    for name, record in packaging['files'].items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid packaged asset name')
        path = a.bundle/name
        if path.stat().st_size != record['bytes'] or file_sha(path) != record['sha256']:
            raise ValueError('Packaged model asset differs')
        files['committee/'+name] = path.read_bytes()
    with zipfile.ZipFile(a.wheel) as wheel:
        metadata = [n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError('Unexpected wheel metadata')
        versions = [l.removeprefix('Version: ') for l in wheel.read(metadata[0]).decode().splitlines() if l.startswith('Version: ')]
        if len(versions) != 1:
            raise ValueError('Missing wheel version')
    files[a.wheel.name] = a.wheel.read_bytes(); files['LICENSE'] = Path('LICENSE').read_bytes()
    files['packaging-report.json'] = a.packaging_report.read_bytes()
    files['installed-bundle-verification.json'] = a.runtime_report.read_bytes()
    for lang, text in (('ro', 'Am nevoie de ajutor cu tema.'), ('uk', 'Мені потрібна допомога з домашнім завданням.')):
        files['example-'+lang+'.json'] = (json.dumps({'language': lang, 'turns': [{'role': 'minor', 'text': text}]}, ensure_ascii=False)+'\n').encode()
    files['README.txt'] = (
        'RoGuard experimental binary review committee\n\n'
        'UNPUBLISHED ENGINEERING ARTIFACT. This is not a qualified v10 release.\n'
        'The local wheel retains distribution version '+versions[0]+'. Do not confuse it with the published stable wheel.\n'
        'It contains the experimental shared-encoder committee and its checked loader.\n'
        'All six adapter files are included. The pretrained Gemma base is not included.\n'
        'Inference requires the existing Apple Silicon MLX environment and the pinned\n'
        'mlx-community/gemma-4-e4b-it-4bit base revision\n'
        '475b9088d29754a3379866cf5aeb6b41acd313c2. No downloads occur in this loader.\n\n'
        'On Apple Silicon, create a separate Python 3.12 environment and install:\n'
        'python -m pip install "./'+a.wheel.name+'[gemma4]"\n'
        'The inference command is:\n'
        'python -m roguard.committee_bundle --bundle committee/committee.json '
        '--sha256 '+packaging['manifest_sha256']+' '
        '--base /path/to/the/pinned/base --input example-ro.json\n\n'
        'The JSON input declares language ro or uk and a list of role/text turns.\n'
        'The focal turn must be a minor message; the native limit is 12 turns and\n'
        '1024 model tokens. See the repository for the separate longer-input workflow.\n'
        'Outputs expose all component decisions and disagreement. A negative result\n'
        'does not establish safety. No result authorizes notification or external action.\n\n'
        'This bundle reproduces the exposed binary engineering comparison. Semantic\n'
        'sets, response checking, fresh calibration, independent evaluation and measured\n'
        'reviewer benefit remain outside this artifact and are required for full v10 qualification.\n'
    ).encode()
    import hashlib
    inventory = {name: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()} for name, value in sorted(files.items())}
    files['archive-manifest.json'] = (json.dumps({'format': 'roguard-committee-research-archive-1',
        'qualified': False, 'public_package_release': False, 'files': inventory}, indent=2)+'\n').encode()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as archive:
            for name, data in sorted(files.items()):
                member = tarfile.TarInfo('roguard-committee-research/'+name)
                member.size = len(data); member.mode = 0o644; member.mtime = 0
                archive.addfile(member, io.BytesIO(data))
    with tarfile.open(a.output, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(files) or any(not member.isfile() for member in members):
            raise ValueError('Unexpected archive entry')
        for member in members:
            name = member.name.removeprefix('roguard-committee-research/')
            if name not in files or archive.extractfile(member).read() != files[name]:
                raise ValueError('Archive roundtrip bytes differ')
    report = {'complete': True, 'source_sha256': file_sha(Path(__file__)), 'archive_sha256': file_sha(a.output),
        'archive_bytes': a.output.stat().st_size, 'files_verified': len(files), 'wheel_version': versions[0],
        'wheel_sha256': file_sha(a.wheel), 'bundle_manifest_sha256': packaging['manifest_sha256'],
        'packaging_report_sha256': file_sha(a.packaging_report), 'runtime_report_sha256': file_sha(a.runtime_report),
        'all_archive_bytes_verified': True, 'public_package_release': False, 'stable_release_ready': False}
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
