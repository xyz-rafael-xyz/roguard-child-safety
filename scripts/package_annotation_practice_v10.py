"""Package language-specific practice forms, keeping coordinator keys separate."""
import argparse
from importlib.resources import files
import json
from pathlib import Path
import zipfile
import hashlib

from roguard.trajectory_annotation import render
from roguard.trajectory_data import file_sha, read_jsonl
from roguard.trajectory_review import annotation_packet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'coordinator', 'output', 'report'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if any(path.exists() for path in (a.coordinator, a.output, a.report)):
        raise ValueError('Refusing to overwrite practice material')
    cases = read_jsonl(a.cases)
    if not cases or any(row.get('synthetic') is not True or row.get('annotation_basis') != 'developer-authored-unreviewed' for row in cases):
        raise ValueError('This packager is only for explicitly synthetic practice cases')
    if {row['language'] for row in cases} != {'ro', 'uk'}:
        raise ValueError('Expected bilingual practice material')
    a.coordinator.mkdir(parents=True); contents = {}; packets = {}
    for lang in ('ro', 'uk'):
        for participant in ('a', 'b'):
            identifier = 'practice-'+lang+'-'+participant
            packet, key = annotation_packet([r for r in cases if r['language'] == lang], annotator_id=identifier)
            (a.coordinator/(identifier+'.json')).write_text(json.dumps(packet, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            (a.coordinator/(identifier+'-PRIVATE-key.json')).write_text(json.dumps(key, indent=2)+'\n', encoding='utf-8')
            contents[identifier+'.html'] = render(packet).encode('utf-8')
            packets[identifier] = {'language': lang, 'items': len(packet['items']), 'packet_digest': key['packet_digest']}
    contents['README.txt'] = (
        'RoGuard annotation practice\n\n'
        'These are exposed developer-authored synthetic exercises. They are not\n'
        'an independent final evaluation and cannot qualify v10 for release.\n\n'
        'Coordinator: give each annotator only their assigned language/letter HTML.\n'
        'The a and b forms contain the same cases in different blinded assignments.\n'
        'Keep coordinator packets and PRIVATE-key files separately; they are not\n'
        'included in this archive. Agree on a written review policy before labeling.\n'
        'Use practice to clarify disagreements; keep real study cases separate.\n\n'
        'Annotator: open your HTML locally. Work independently, without model\n'
        'answers. Label only the displayed prefix. Mark uncertainty explicitly.\n'
        'Select supporting text within a message and press Add selected text.\n'
        'Download a draft before closing; answers are not automatically saved.\n'
        'Use Resume with that draft to continue. Download completed annotations\n'
        'and return the JSON file to your coordinator when finished.\n\n'
        'The form makes no network requests. Automated tests cover export,\n'
        'resumption and evidence offsets; real-browser visual verification is\n'
        'still pending. Report interface problems during practice.\n'
    ).encode('utf-8')
    inventory = {name: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()} for name, value in contents.items()}
    manifest = {'format': 'roguard-annotation-practice-1', 'practice_only': True,
                'human_annotations_obtained': False, 'real_browser_verified': False,
                'packets': packets, 'files': inventory}
    contents['manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode('utf-8')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value)
    with zipfile.ZipFile(a.output) as archive:
        if set(archive.namelist()) != set(contents) or any(archive.read(name) != value for name, value in contents.items()):
            raise ValueError('Practice archive roundtrip differs')
    resource = files('roguard')
    report = {'complete': True, 'source_sha256': file_sha(__file__), 'cases_sha256': file_sha(a.cases),
              'archive_sha256': file_sha(a.output), 'archive_bytes': a.output.stat().st_size,
              'files_verified': len(contents), 'coordinator_keys_in_archive': False,
              'render_sources': {name: hashlib.sha256(resource.joinpath(name).read_bytes()).hexdigest()
                                 for name in ('trajectory_annotation.py', 'trajectory-annotation.html', 'trajectory-annotation.js')},
              'packets': packets, 'practice_only': True, 'human_annotations_obtained': False,
              'real_browser_verified': False, 'stable_release_ready': False}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
