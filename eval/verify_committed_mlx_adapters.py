"""Verify selected private MLX artifacts and optionally replay one saved card each."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.review import CATEGORIES, load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "ro-v4": ("batch-0012", "eval/runs/ro-v4-test-0012.json", "romistral"),
    "ro-v6": ("batch-0016", "eval/runs/ro-v6-test-0016.json", "romistral"),
    "ro-qwen-v7": ("batch-0018", "eval/runs/ro-qwen-v7-wrapper-test-0018.json", "qwen"),
    "ro-qwen-v8": ("batch-0018", "eval/runs/ro-qwen-v8-test-0018.json", "qwen"),
}


def verify(root: Path = ROOT, base_paths: dict[str, Path] | None = None) -> dict:
    results = {}
    for study, (batch, run_path, base_kind) in SPECS.items():
        artifact = root / "models" / f"{study}-abstract"
        manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
        metadata = json.loads((artifact / "roguard_metadata.json").read_text(encoding="utf-8"))
        config = json.loads((artifact / "adapter_config.json").read_text(encoding="utf-8"))
        selection_path = root / "eval/runs" / f"{study}-dev-selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        run = json.loads((root / run_path).read_text(encoding="utf-8"))
        if (manifest.get("study") != study or
                manifest.get("artifact_kind") != "experimental_romanian_abstract_card_mlx_adapter" or
                manifest.get("adapter_weight_sha256") != sha256(artifact / "adapters.safetensors") or
                manifest.get("adapter_weight_sha256") != selection["selected_weight_sha256"] or
                manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
                manifest.get("metadata_sha256") != sha256(artifact / "roguard_metadata.json") or
                manifest.get("selection_record_sha256") != sha256(selection_path) or
                manifest.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v4.py") or
                manifest.get("base_model") != metadata["base_model_id"] or
                manifest.get("base_revision") != metadata["base_revision"] or
                metadata.get("base_model_path") is not None or
                metadata.get("selected_step") != selection["selected_step"] or
                metadata.get("prompt_version") != "v4" or
                config != {"fine_tune_type": "lora", "num_layers": 8,
                           "lora_parameters": {"rank": 16, "dropout": 0.05, "scale": 2.0}} or
                run.get("batches") != [batch] or not run.get("predictions")):
            raise ValueError(f"Private {study} adapter differs from its frozen record")
        reproduced = None
        if base_paths is not None:
            from roguard import MLXClassifier, screen
            from roguard.qwen_output import parse_qwen_binary
            base = base_paths.get(base_kind)
            if base is None:
                raise ValueError(f"Missing {base_kind} local base path")
            rows = load_approved(root, [batch])
            prior = run["predictions"][0]
            backend = MLXClassifier(artifact, base_model_path=base,
                                    binary_parser=parse_qwen_binary if base_kind == "qwen" else None)
            current = screen(rows[0]["text"], language="ro",
                             source_kind=rows[0]["source_kind"], backend=backend,
                             thresholds={code: 0.5 for code in CATEGORIES})
            reproduced = prior["id"] == rows[0]["id"] and list(current.labels) == prior["predicted"]
            if not reproduced:
                raise ValueError(f"Private {study} adapter differs on a saved symbolic card")
        results[study] = {"weight_sha256": manifest["adapter_weight_sha256"],
                          "saved_label_reproduced": reproduced}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--romistral-base", type=Path)
    parser.add_argument("--qwen-base", type=Path)
    args = parser.parse_args()
    paths = None if args.romistral_base is None and args.qwen_base is None else {
        "romistral": args.romistral_base, "qwen": args.qwen_base}
    print(json.dumps(verify(base_paths=paths), indent=2))
