"""Export only attested abstract cards to MLX prompt/completion files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.prompt import format_codes, make_prompt, make_prompt_v2, make_prompt_v3
from roguard.prompt_v4 import applicable_codes, make_prompt_v4
from roguard.review import load_train_dev, sha256, taxonomy_sha256


def prepare(root: Path, batch_ids: list[str], language: str, output: Path, prompt_version: str = "v1") -> dict:
    if prompt_version not in ("v1", "v2", "v3", "v4"):
        raise ValueError("Unknown prompt version")
    prompt_fn = {"v1": make_prompt, "v2": make_prompt_v2, "v3": make_prompt_v3}.get(prompt_version)
    train, dev = load_train_dev(root, batch_ids, language)
    output = output.resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("Output path must be a directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    task_counts = {}
    for filename, rows in (("train.jsonl", train), ("valid.jsonl", dev)):
        if prompt_version == "v4":
            tasks = []
            for row in rows:
                for code in applicable_codes(row["source_kind"]):
                    task = {"prompt": make_prompt_v4(row, code),
                            "completion": "da" if code in row["labels"] else "nu"}
                    repeats = 3 if filename == "train.jsonl" and row["source_kind"] == "response" and code in row["labels"] else 1
                    tasks.extend([task] * repeats)
        else:
            tasks = [{"prompt": prompt_fn(row), "completion": format_codes(row["labels"])} for row in rows]
        (output / filename).write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8")
        task_counts[filename] = len(tasks)
    manifest = {
        "language": language, "batches": batch_ids,
        "taxonomy_sha256": taxonomy_sha256(root, language),
        "train_rows": len(train), "dev_rows": len(dev),
        "train_tasks": task_counts["train.jsonl"], "dev_tasks": task_counts["valid.jsonl"],
        "train_sha256": sha256(output / "train.jsonl"),
        "valid_sha256": sha256(output / "valid.jsonl"),
        "format": "mlx_lm_binary_v1" if prompt_version == "v4" else "mlx_lm_completions_v1",
        "prompt_version": prompt_version,
        "synthetic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", action="append", required=True)
    parser.add_argument("--language", choices=("ro",), required=True)
    parser.add_argument("--prompt-version", choices=("v1", "v2", "v3", "v4"), default="v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.batch, args.language, args.output, args.prompt_version), indent=2))


if __name__ == "__main__":
    main()
