"""Prepare fresh cache workspaces using only the public corpus metadata.

This copies metadata, never private tensor caches or model weights. The optional
extraction runs the existing pinned frozen-body extractor sequentially.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from roguard.trajectory_data import file_sha

CORPORA = ('shared-tail-corpus-164', 'grounding-tail-corpus-174',
           'referent-tail-corpus-176', 'reported-tail-corpus-195')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',type=Path,help='When supplied, also extract all four corpora; requires substantial disk and Apple Silicon.')
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite a cache workspace')
    sources={}
    for name in CORPORA:
        root=Path('research/v9-stabilization')/name
        for path,expected in json.loads((root/'source-freeze.json').read_text()).items():
            if file_sha(path)!=expected:raise ValueError('Public frozen input differs: '+path)
        required={'inputs.jsonl.gz','source-freeze.json','base-manifest.json','software.json','protocol.json','common-prefix.json'}
        sources[name]=[root/filename for filename in sorted(required)]
        if any(not p.is_file() for p in sources[name]):raise ValueError('Missing public cache metadata')
    a.output.mkdir(parents=True)
    manifest={'format':'roguard-public-tail-cache-workspace-1','source_sha256':file_sha(Path(__file__)),
              'private_weights_or_caches_copied':False,'corpora':{}}
    for name,paths in sources.items():
        destination=a.output/name;destination.mkdir()
        manifest['corpora'][name]={}
        for source in paths:
            shutil.copyfile(source,destination/source.name)
            manifest['corpora'][name][source.name]=file_sha(destination/source.name)
    (a.output/'preparation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if a.base is not None:
        for name in CORPORA:
            subprocess.run([sys.executable,'-m','training.extract_additional_tail_corpus_v9',
                            '--base',str(a.base),'--study',str(a.output/name)],check=True)
    print(json.dumps({'corpora':list(CORPORA),'extraction_requested':a.base is not None,
                      'private_weights_or_caches_copied':False},indent=2))


if __name__=='__main__':main()
