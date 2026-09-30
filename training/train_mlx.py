"""Run local QLoRA only after every selected abstract batch is attested."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from roguard.review import CATEGORIES, load_train_dev
from convert_base import MODEL_ID, file_digest
from prepare_mlx import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", action="append", required=True)
    parser.add_argument("--model", type=Path, required=True, help="Local 4-bit MLX RoMistral directory")
    parser.add_argument("--base-revision", required=True, help="Exact upstream Hugging Face commit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--prompt-version", choices=("v1", "v2", "v3", "v4"), default="v1")
    args = parser.parse_args()
    if args.iters < 1:
        raise ValueError("Training iterations must be positive")
    if len(args.base_revision) != 40 or any(char not in "0123456789abcdef" for char in args.base_revision):
        raise ValueError("Use a 40-character lowercase commit revision")
    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("Output path must be a directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")

    # Review gate runs before model preflight, any output, or training.
    load_train_dev(args.root, args.batch, "ro")
    model_path = args.model.resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Pass a converted local MLX model directory")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if model_config.get("model_type") != "mistral" or not model_config.get("quantization"):
        raise ValueError("Expected a quantized local Mistral model")
    provenance_path = model_path / "roguard_base.json"
    if not provenance_path.is_file():
        raise ValueError("Local base model lacks verified RoGuard provenance")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("source_model") != MODEL_ID or provenance.get("source_revision") != args.base_revision:
        raise ValueError("Base model revision or source differs from requested RoMistral")
    if provenance.get("config_sha256") != file_digest(config_path):
        raise ValueError("Base model configuration changed after verification")
    for name, digest in provenance.get("quantized_weight_sha256", {}).items():
        if file_digest(model_path / name) != digest:
            raise ValueError(f"Quantized weight changed after verification: {name}")
    if not provenance.get("quantized_weight_sha256"):
        raise ValueError("No verified quantized weights")
    manifest = prepare(args.root, args.batch, "ro", output / "data", args.prompt_version)
    adapter = output / "adapter"
    command = [
        str(Path(sys.executable).parent / "mlx_lm.lora"),
        "--model", str(model_path), "--train", "--data", str(output / "data"),
        "--adapter-path", str(adapter), "--config", str(args.root / "training" / "mlx_lora_config.yaml"),
        "--mask-prompt", "--num-layers", "8", "--batch-size", "1",
        "--iters", str(args.iters), "--learning-rate", "1e-5", "--max-seq-length", "512",
        "--grad-checkpoint", "--steps-per-eval", "50", "--val-batches", "8",
        "--save-every", "100", "--seed", "20260929",
    ]
    subprocess.run(command, check=True)
    (adapter / "roguard_metadata.json").write_text(json.dumps({
        "backend": "mlx_lm_codes", "base_model_path": str(model_path),
        "base_model_id": "OpenLLM-Ro/RoMistral-7b-Instruct", "base_revision": args.base_revision,
        "language": "ro", "categories": CATEGORIES, "batches": args.batch,
        "taxonomy_sha256": manifest["taxonomy_sha256"],
        "train_sha256": manifest["train_sha256"], "valid_sha256": manifest["valid_sha256"],
        "train_rows": manifest["train_rows"], "dev_rows": manifest["dev_rows"],
        "train_tasks": manifest["train_tasks"], "dev_tasks": manifest["dev_tasks"],
        "prompt_version": args.prompt_version,
        "iters": args.iters, "synthetic_only": True,
        "score_semantics": "hard_code_output_not_calibrated_probability",
    }, indent=2) + "\n", encoding="utf-8")
    print(adapter)


if __name__ == "__main__":
    main()
