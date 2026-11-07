"""Package the recorded binary engineering committee into a new local bundle."""
import argparse
import json
from pathlib import Path

from roguard.committee_bundle import export_committee_bundle, load_committee
from roguard.trajectory_data import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime_study', 'readout_artifacts', 'attention_artifacts', 'base', 'output', 'report'):
        parser.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('Refusing to overwrite packaging verification')
    runtime = json.loads((args.runtime_study/'report.json').read_text())
    protocol = json.loads((args.runtime_study/'protocol.json').read_text())
    if not runtime['complete'] or runtime['decision'] != 'BINARY_RUNTIME_ENGINEERING_PASS_ONLY':
        raise ValueError('The fixed committee must pass fresh runtime verification before packaging')
    root = Path('research/v10-evidence'); models = {}
    for lang in ('ro', 'uk'):
        models[lang] = {
            'v9': (args.runtime_study/'v9-binding'/(lang+'-review.json'),
                   Path('research/v9-stabilization/reported-tail-fit-196/actual')/(lang+'-selected.safetensors')),
            'readout': (root/'semantic-readout-11'/(lang+'-review.json'), args.readout_artifacts/(lang+'-selected.safetensors')),
            'attention': (root/'semantic-tail-14'/(lang+'-review.json'), args.attention_artifacts/(lang+'-selected.safetensors'))}
    path, sha = export_committee_bundle(args.output, models)
    identities = {}
    for lang in ('ro', 'uk'):
        model = load_committee(path, expected_sha256=sha, base=args.base, language=lang)
        if model.identity != protocol['committee_identities'][lang]:
            raise ValueError('Packaged committee identity differs from the runtime evaluation')
        identities[lang] = model.identity
    report = {'complete': True, 'format': 'roguard-committee-packaging-check-1',
        'runtime_report_sha256': file_sha(args.runtime_study/'report.json'),
        'source_sha256': file_sha(Path(__file__)), 'manifest_sha256': sha,
        'files': {p.name: {'sha256': file_sha(p), 'bytes': p.stat().st_size} for p in sorted(args.output.iterdir())},
        'committee_identities': identities, 'pinned_base_included': False,
        'qualification': 'binary engineering component only', 'stable_release_ready': False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
