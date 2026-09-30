"""Fetch or locate the pinned multilingual NLI base used by V21."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from roguard.nli_v21 import BASE_HASHES, MODEL_ID, REVISION, verify_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Use only an already cached snapshot")
    args = parser.parse_args()
    snapshot = Path(snapshot_download(
        MODEL_ID, revision=REVISION, local_files_only=args.offline,
        allow_patterns=tuple(BASE_HASHES)))
    verify_base(snapshot)
    print(snapshot)


if __name__ == "__main__":
    main()
