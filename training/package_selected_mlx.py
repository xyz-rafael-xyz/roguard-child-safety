"""Package hash-selected failed MLX comparators for the private repository."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ("ro-v4", "ro-v6", "ro-qwen-v7", "ro-qwen-v8")


def package(study: str, root: Path = ROOT) -> Path:
    if study not in STUDIES:
        raise ValueError("Unknown selected MLX study")
    source = root / "checkpoints" / study / "selected-adapter"
    selection_path = root / "eval/runs" / f"{study}-dev-selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_weight = source / "adapters.safetensors"
    original_config = source / "adapter_config.json"
    original_metadata = source / "roguard_metadata.json"
    metadata = json.loads(original_metadata.read_text(encoding="utf-8"))
    config = json.loads(original_config.read_text(encoding="utf-8"))
    prompt_sha = sha256(root / "src/roguard/prompt_v4.py")
    if (selection["selected_weight_sha256"] != sha256(source_weight) or
            (selection.get("prompt_sha256") not in (None, prompt_sha)) or
            metadata.get("selected_step") != selection["selected_step"] or
            metadata.get("prompt_version") != "v4" or
            config.get("fine_tune_type") != "lora" or
            config.get("num_layers") != 8 or
            not isinstance(config.get("lora_parameters"), dict)):
        raise ValueError("Selected MLX artifact differs from development record")
    target = root / "models" / f"{study}-abstract"
    if target.exists():
        raise FileExistsError("Preserve an existing packaged adapter")
    target.mkdir(parents=True)
    shutil.copyfile(source_weight, target / "adapters.safetensors")
    portable_config = {key: config[key] for key in
                       ("fine_tune_type", "num_layers", "lora_parameters")}
    (target / "adapter_config.json").write_text(
        json.dumps(portable_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata["base_model_path"] = None
    (target / "roguard_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "artifact_kind": "experimental_romanian_abstract_card_mlx_adapter",
        "study": study, "base_model": metadata["base_model_id"],
        "base_revision": metadata["base_revision"],
        "adapter_weight_sha256": sha256(target / "adapters.safetensors"),
        "adapter_config_sha256": sha256(target / "adapter_config.json"),
        "metadata_sha256": sha256(target / "roguard_metadata.json"),
        "selection_record_sha256": sha256(selection_path),
        "original_adapter_config_sha256": sha256(original_config),
        "original_metadata_sha256": sha256(original_metadata),
        "prompt_sha256": prompt_sha,
        "evaluation_status": "failed_registered_synthetic_target",
        "trained_language": "ro", "trained_domain": "abstract_symbolic_cards_only",
    }
    (target / "research.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=STUDIES)
    args = parser.parse_args()
    print(package(args.study))
