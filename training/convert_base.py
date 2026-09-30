"""Verify pinned RoMistral weights, then convert locally to 4-bit MLX."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

MODEL_ID = "OpenLLM-Ro/RoMistral-7b-Instruct"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Pinned hf download directory")
    parser.add_argument("--revision", required=True, help="40-character Hugging Face commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(char not in "0123456789abcdef" for char in args.revision):
        raise ValueError("Use a 40-character lowercase commit revision")
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Conversion output must not exist")
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL_ID, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("Hugging Face did not resolve the requested revision")
    expected = {entry.rfilename: entry.lfs.sha256 for entry in info.siblings
                if entry.rfilename.endswith(".safetensors") and entry.lfs}
    if not expected:
        raise ValueError("No source weight hashes in model metadata")
    for name, digest in expected.items():
        path = source / name
        if not path.is_file() or file_digest(path) != digest:
            raise ValueError(f"Source weight missing or differs from pinned revision: {name}")
    config_path = source / "config.json"
    if json.loads(config_path.read_text(encoding="utf-8")).get("model_type") != "mistral":
        raise ValueError("Expected Mistral source configuration")
    command = [str(Path(sys.executable).parent / "mlx_lm.convert"),
               "--hf-path", str(source), "--mlx-path", str(output), "-q", "--q-bits", "4"]
    subprocess.run(command, check=True)
    quantized = sorted(output.glob("*.safetensors"))
    if not quantized:
        raise ValueError("Conversion produced no weights")
    marker = {"source_model": MODEL_ID, "source_revision": args.revision,
              "source_weight_sha256": expected,
              "quantized_weight_sha256": {path.name: file_digest(path) for path in quantized},
              "config_sha256": file_digest(output / "config.json"),
              "converter": "mlx_lm.convert", "quantization_bits": 4}
    (output / "roguard_base.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
