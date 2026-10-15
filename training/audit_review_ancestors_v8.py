"""Check recorded v7 and semantic-v8 ancestor training against current evaluation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from roguard.context import render
from roguard.semantic import normalize_semantic_text


def load(path):
    raw=path.read_bytes()
    return [json.loads(line) for line in (gzip.decompress(raw) if path.suffix=='.gz' else raw).decode().splitlines()]


def key(row):
    return row['language'],normalize_semantic_text(render(row['turns'][:row.get('focus',len(row['turns'])-1)+1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths=sorted(Path('research/v7-1-context').glob('candidate-*/selected-training.jsonl'))
    paths += [Path(f'research/v8-evidence/{phase}/selected-training.jsonl.gz') for phase in ('semantic-04','semantic-05')]
    history=set();sources={}
    for path in paths:
        rows=load(path);contexts={key(r) for r in rows};history|=contexts
        sources[str(path)]={'rows':len(rows),'unique_contexts':len(contexts),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    overlaps={split:[r['case_id'] for r in load(a.candidate/(split+'.jsonl.gz')) if key(r) in history] for split in ('dev','cal')}
    result={'candidate':str(a.candidate),'scope':'Recorded v7 candidate training superset plus semantic04/05 training; checks normalized context inputs, not semantic-family independence','ancestors':sources,'unique_historical_contexts':len(history),'overlaps':overlaps}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'historical_contexts':len(history),'overlaps':overlaps}))
    raise SystemExit(int(any(overlaps.values())))

if __name__=='__main__':main()
