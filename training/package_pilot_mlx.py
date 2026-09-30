"""Preserve selected historical MLX pilots without copying intermediate checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from roguard.review import CATEGORIES, sha256

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "ro-v1": ("v1", 200, "batch-0007", "eval/runs/ro-v1.json", None),
    "ro-v2": ("v2", 200, "batch-0008", "eval/runs/ro-v2-test-0008.json",
              "eval/runs/ro-v2-test-0008-manifest.json"),
    "ro-v3": ("v3", 600, "batch-0011", "eval/runs/ro-v3-test-0011.json",
              "eval/runs/ro-v3-test-0011-comparison.json"),
}


def package(study: str, root: Path = ROOT) -> Path:
    if study not in SPECS:
        raise ValueError("Unknown historical MLX pilot")
    prompt_version, selected_step, batch, run_name, record_name = SPECS[study]
    source = root / "checkpoints" / study / "adapter"
    weight = source / "adapters.safetensors"
    config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((source / "roguard_metadata.json").read_text(encoding="utf-8"))
    run = json.loads((root / run_name).read_text(encoding="utf-8"))
    weight_hash = sha256(weight)
    step_weight = source / f"{selected_step:07d}_adapters.safetensors"
    if (weight_hash != sha256(step_weight) or
            metadata.get("prompt_version", "v1") != prompt_version or
            metadata.get("iters") != selected_step or
            tuple(metadata.get("categories", ())) != CATEGORIES or
            metadata.get("language") != "ro" or
            metadata.get("base_model_id") != "OpenLLM-Ro/RoMistral-7b-Instruct" or
            config.get("fine_tune_type") != "lora" or
            config.get("num_layers") != 8 or
            config.get("lora_parameters") != {"rank": 16, "dropout": 0.05, "scale": 2.0} or
            run.get("adapter") != f"checkpoints/{study}/adapter" or
            run.get("batches") != [batch] or not run.get("predictions")):
        raise ValueError("Historical pilot differs from its final step or saved run")
    if record_name:
        record = json.loads((root / record_name).read_text(encoding="utf-8"))
        if record.get("adapter_weight_sha256") != weight_hash:
            raise ValueError("Historical pilot weight differs from frozen comparison")
    target = root / "models" / f"{study}-abstract"
    if target.exists():
        raise FileExistsError("Preserve an existing packaged pilot")
    target.mkdir(parents=True)
    shutil.copyfile(weight, target / "adapters.safetensors")
    portable_config = {key: config[key] for key in
                       ("fine_tune_type", "num_layers", "lora_parameters")}
    (target / "adapter_config.json").write_text(
        json.dumps(portable_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata["base_model_path"] = None
    metadata["prompt_version"] = prompt_version
    (target / "roguard_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "artifact_kind": "experimental_romanian_abstract_card_mlx_adapter",
        "study": study,
        "prompt_version": prompt_version,
        "prompt_sha256": sha256(root / "src/roguard/prompt.py"),
        "base_model": metadata["base_model_id"],
        "base_revision": metadata["base_revision"],
        "adapter_weight_sha256": sha256(target / "adapters.safetensors"),
        "adapter_config_sha256": sha256(target / "adapter_config.json"),
        "metadata_sha256": sha256(target / "roguard_metadata.json"),
        "saved_run_path": run_name,
        "saved_run_sha256": sha256(root / run_name),
        "comparison_record_path": record_name,
        "comparison_record_sha256": sha256(root / record_name) if record_name else None,
        "selected_step": selected_step,
        "evaluation_status": "historical_failed_synthetic_pilot",
        "trained_language": "ro",
        "trained_domain": "abstract_symbolic_cards_only",
    }
    (target / "research.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        f"# {study} historical RoMistral pilot\n\n"
        "Private research artifact. Selected final-step LoRA weights for abstract Romanian "
        "cards only; the frozen pilot failed to establish reliable category decisions. "
        "This is not a detector for real child messages. The separate pinned RoMistral "
        "base is required to run it. Its noncommercial license applies independently "
        "of the repository's software license.\n\n"
        f"Saved run: [`{run_name}`](../../{run_name}). "
        "Verify with `python eval/verify_committed_pilot_adapters.py`.\n",
        encoding="utf-8",
    )
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=SPECS)
    args = parser.parse_args()
    print(package(args.study))
