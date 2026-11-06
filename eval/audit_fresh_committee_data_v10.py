"""Exact-prefix and focal-text screen against explicitly enumerated old data."""
import argparse
import json
from pathlib import Path
import re
import unicodedata

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def keys(row):
    focuses = [p['focus'] for p in row['checkpoints']] if 'checkpoints' in row else [row.get('focus', len(row['turns'])-1)]
    for focus in focuses:
        turns = row['turns'][:focus+1]
        yield (digest([row['language'], [(t['role'], normalized(t['text'])) for t in turns]]),
               digest([row['language'], normalized(turns[-1]['text'])]))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite overlap screen')
    paths = sorted(Path('research/v9-stabilization/paired-scope-117/data').glob('*.jsonl'))
    paths += sorted(Path('research/v10-evidence/data').glob('*.jsonl'))
    paths += [Path('research/v10-evidence/broad-v9-regression/cases.jsonl'), Path('research/v10-evidence/scope-input-audit-20/train.jsonl')]
    prefixes = set(); texts = set(); sources = {}; count = 0
    for path in paths:
        rows = read_jsonl(path); n = 0
        for row in rows:
            for prefix, focal in keys(row):
                prefixes.add(prefix); texts.add(focal); count += 1; n += 1
        sources[str(path)] = {'sha256': file_sha(path), 'checkpoints': n}
    hits = []
    for row in read_jsonl(a.data):
        for prefix, focal in keys(row):
            if prefix in prefixes or focal in texts:
                hits.append({'case_id': row['case_id'], 'prefix_overlap': prefix in prefixes, 'focal_text_overlap': focal in texts})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'complete': True, 'source_sha256': file_sha(__file__), 'data_sha256': file_sha(a.data),
        'sources': sources, 'historical_checkpoints': count, 'overlaps': hits,
        'scope': 'Only enumerated datasets; not an exhaustive pretraining or full historical fine-tuning audit.',
        'independent_annotation': False}, indent=2)+'\n')
    print(json.dumps({'historical_checkpoints': count, 'overlaps': len(hits)}))


if __name__ == '__main__':
    main()
