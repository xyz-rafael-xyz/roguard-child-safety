"""Freeze per-language choices using identical eligible development populations."""
import argparse
import json
from pathlib import Path

from roguard.trajectory_data import file_sha, read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attention-study', type=Path, required=True)
    p.add_argument('--reference-study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite specialist selection')
    sources = {'attention': a.attention_study, 'reference': a.reference_study}
    reports = {name: json.loads((path/'report.json').read_text()) for name, path in sources.items()}
    if any(not r['complete'] for r in reports.values()):
        raise ValueError('All candidates must have completed development selection')
    result = {'format': 'roguard-language-specialists-selection-1',
        'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-16.json'),
        'parent_report_sha256': {name: file_sha(path/'report.json') for name, path in sources.items()},
        'languages': {}, 'regression_loaded_for_selection': False, 'stable_release_ready': False}
    for lang in ('ro', 'uk'):
        populations = {name: read_jsonl(path/(lang+'-development-predictions.jsonl')) for name, path in sources.items()}
        identities = [{r['case_id']: (r['gold'], r['group']) for r in rows} for rows in populations.values()]
        if identities[0] != identities[1] or any(len(rows) != len(identities[0]) for rows in populations.values()):
            raise ValueError('Development populations differ')
        ranks = {name: r['languages'][lang]['selected']['selection_rank'] for name, r in reports.items()}
        choice = min(ranks, key=ranks.get)
        suffix = '-review.json' if choice == 'attention' else '-head.json'
        spec_path = sources[choice]/(lang+suffix)
        spec = json.loads(spec_path.read_text())
        result['languages'][lang] = {'selected': choice, 'development_rows': len(identities[0]),
            'candidate_ranks': ranks, 'spec_sha256': file_sha(spec_path),
            'artifact_sha256': spec['adapter_sha256' if choice == 'attention' else 'head_sha256']}
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
