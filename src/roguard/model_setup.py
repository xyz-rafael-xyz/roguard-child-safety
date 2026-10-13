"""Explicit opt-in download of the pinned optional semantic base model."""
from __future__ import annotations
import argparse
import json
from .mmbert_study import MODEL_ID,REVISION,BASE_HASHES,verify_base
from pathlib import Path

def locate_model():
    """Resolve cached pinned files only. Inference never downloads implicitly."""
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    except (ImportError,OSError,ValueError) as exc:
        raise ValueError('Install the semantic extra and run roguard-model fetch first') from exc


def main():
    parser=argparse.ArgumentParser(description='Fetch or locate the optional pinned semantic model')
    parser.add_argument('action',choices=['fetch','locate'])
    args=parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
        path=Path(snapshot_download(MODEL_ID,revision=REVISION,local_files_only=args.action=='locate',
                                   allow_patterns=list(BASE_HASHES)+['special_tokens_map.json','README.md']))
        verify_base(path)
        print(path)
    except ImportError:parser.error('Install the semantic extra first: pip install "roguard-child-safety[semantic]"')
    except (OSError,ValueError):parser.error('Model unavailable or digest mismatch; fetch again explicitly')

if __name__=='__main__':main()
