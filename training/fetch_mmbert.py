"""Fetch or locate the pinned mmBERT snapshot and verify its study files."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from roguard.mmbert_study import BASE_HASHES, MODEL_ID, REVISION, verify_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Use only an already cached snapshot")
    args = parser.parse_args()
    snapshot = Path(snapshot_download(
        MODEL_ID, revision=REVISION, local_files_only=args.offline,
        allow_patterns=tuple(BASE_HASHES) + ("special_tokens_map.json",)))
    verify_base(snapshot)
    print(snapshot)


if __name__ == "__main__":
    main()
