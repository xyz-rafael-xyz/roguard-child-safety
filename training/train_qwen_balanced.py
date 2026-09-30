"""Fit pinned Qwen3 with equal exposure to all six Romanian category tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from roguard.review import CATEGORIES, load_train_dev, sha256
from prepare_balanced_mlx import prepare_balanced

MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
BASE_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
EXPECTED_BASE_HASHES = {
    "config.json": "574349e5a343236546fda55e4744a76e181f534182d7dc60ff1bad7e7a502849",
    "model.safetensors": "2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "4397cc477eb6d79715ccd2000accd6b3531928f30029665832fa1b255f24d2b9",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true",
                        help="Fetch the pinned model snapshot when it is not already cached")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Transfer output directory must be empty")
    batches = ["batch-0014", "batch-0015"]
    load_train_dev(root, batches, "ro")
    model = Path(snapshot_download(
        MODEL_ID, revision=REVISION, local_files_only=not args.download,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt"],
    ))
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3" or config.get("quantization", {}).get("bits") != 4:
        raise ValueError("Expected pinned quantized Qwen3 model")
    base_hashes = {name: sha256(model / name) for name in BASE_FILES}
    if base_hashes != EXPECTED_BASE_HASHES:
        raise ValueError("Pinned Qwen3 base files differ from preregistered hashes")
    manifest = prepare_balanced(root, output / "data")
    adapter = output / "adapter"
    command = [
        str(Path(sys.executable).parent / "mlx_lm.lora"),
        "--model", str(model), "--train", "--data", str(output / "data"),
        "--adapter-path", str(adapter), "--config", str(root / "training" / "mlx_lora_config.yaml"),
        "--mask-prompt", "--num-layers", "8", "--batch-size", "1",
        "--iters", "1200", "--learning-rate", "1e-5", "--max-seq-length", "512",
        "--grad-checkpoint", "--steps-per-eval", "50", "--val-batches", "8",
        "--save-every", "100", "--seed", "20260929",
    ]
    subprocess.run(command, check=True)
    (adapter / "roguard_metadata.json").write_text(json.dumps({
        "backend": "mlx_lm_codes", "base_model_path": str(model),
        "base_model_id": MODEL_ID, "base_revision": REVISION, "base_files_sha256": base_hashes,
        "language": "ro", "categories": CATEGORIES, "batches": batches,
        "taxonomy_sha256": manifest["taxonomy_sha256"],
        "train_sha256": manifest["train_sha256"], "valid_sha256": manifest["valid_sha256"],
        "train_rows": manifest["train_rows"], "dev_rows": manifest["dev_rows"],
        "train_tasks": manifest["train_tasks"], "dev_tasks": manifest["dev_tasks"],
        "prompt_version": "v4", "iters": 1200, "synthetic_only": True,
        "training_balance_rule": manifest["balance_rule"],
        "train_tasks_per_category": manifest["train_tasks_per_category"],
        "score_semantics": "hard_code_output_not_calibrated_probability",
    }, indent=2) + "\n", encoding="utf-8")
    print(adapter)


if __name__ == "__main__":
    main()
