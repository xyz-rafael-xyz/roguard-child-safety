"""Reproduce a tokenizer-only Ukrainian candidate audit; never load model weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data" / "synthetic" / "uk-contrast-0001.jsonl"
PROBE_SHA256 = "87e974bf932620f71159c8b4f17cb02dbc9fa2c1c145656771aa6cbc40b28ef2"
CANDIDATES = (
    ("ukr_roberta", "youscan/ukr-roberta-base", "8149bde480a6df9014aa934c3f8af30858dea5a6"),
    ("uk_modernbert", "KoichiYasuoka/modernbert-base-ukrainian", "a64c57e5a16e95e27b09227cd335bb234be12ea1"),
    ("mmbert", "jhu-clsp/mmBERT-base", "c5955035435e2bf121cde7f3c8863ef52ff35d82"),
)
TOKENIZER_FILES = (
    "config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
    "vocab.json", "merges.txt", "special_tokens_map.json",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(fetch: bool = False) -> dict:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    source = PROBE.read_bytes()
    if sha256(source) != PROBE_SHA256:
        raise ValueError("Ukrainian diagnostic probe changed")
    rows = [json.loads(line) for line in source.decode("utf-8").splitlines()]
    if len(rows) != 48 or any(row.get("language") != "uk" for row in rows):
        raise ValueError("Expected the 48 historical Ukrainian abstract cards")
    report = {
        "scope": "Tokenizer lengths on a consumed, unreviewed abstract-card probe; no accuracy result",
        "probe_path": str(PROBE.relative_to(ROOT)),
        "probe_sha256": PROBE_SHA256,
        "rows": len(rows),
        "token_limit": 256,
        "toolchain": {
            "python": sys.version.split()[0],
            "transformers": importlib.metadata.version("transformers"),
            "huggingface_hub": importlib.metadata.version("huggingface_hub"),
            "sentencepiece": importlib.metadata.version("sentencepiece"),
            "protobuf": importlib.metadata.version("protobuf"),
        },
        "candidates": [],
    }
    for name, repo_id, revision in CANDIDATES:
        path = Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=TOKENIZER_FILES,
            local_files_only=not fetch,
        ))
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        encoded = tokenizer([row["text"] for row in rows], add_special_tokens=True)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        unknown = tokenizer.unk_token_id
        files = {name: sha256((path / name).read_bytes())
                 for name in TOKENIZER_FILES if (path / name).is_file()}
        report["candidates"].append({
            "name": name,
            "repo_id": repo_id,
            "revision": revision,
            "tokenizer_file_sha256": files,
            "median_tokens": statistics.median(lengths),
            "max_tokens": max(lengths),
            "rows_over_limit": sum(length > 256 for length in lengths),
            "unknown_tokens": sum(ids.count(unknown) for ids in encoded["input_ids"])
            if unknown is not None else None,
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Fetch pinned tokenizer files only")
    args = parser.parse_args()
    print(json.dumps(audit(fetch=args.fetch), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
