"""Development-only diagnostic: compare explicit topic resets with/without prior text."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(l) for l in gzip.decompress((a.candidate/'dev.jsonl.gz').read_bytes()).decode().splitlines()]
    selected=[]
    for row in rows:
        # These generated transfer rows explicitly reset topic; no implicit
        # references or inherited framing are stripped from ordinary inputs.
        transfer=row['case_id'].startswith('native-transfer-dev-')
        v71=row['case_id'].startswith('native-v71-dev-') and row['attributes']['concern']=='1' and row['attributes']['subject']=='SELF' and row['attributes']['time']=='CURRENT'
        if not (transfer or v71) or len(row['turns'])!=2:continue
        original=dict(row);original['case_id']+='-with-prior';selected.append(original)
        isolated=dict(row);isolated['case_id']+='-without-prior';isolated['turns']=[row['turns'][-1]];isolated['focus']=0;selected.append(isolated)
    if not selected:raise ValueError('No explicit development reset pairs')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_bytes(gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected).encode(),mtime=0))
    print(len(selected),'development diagnostic rows; not fresh validation')

if __name__=='__main__':main()
