"""Check historical pilot artifacts against frozen runs; optionally replay one card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.review import CATEGORIES, load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "ro-v1": ("v1", 200, "batch-0007", "eval/runs/ro-v1.json", None),
    "ro-v2": ("v2", 200, "batch-0008", "eval/runs/ro-v2-test-0008.json",
              "eval/runs/ro-v2-test-0008-manifest.json"),
    "ro-v3": ("v3", 600, "batch-0011", "eval/runs/ro-v3-test-0011.json",
              "eval/runs/ro-v3-test-0011-comparison.json"),
}


def verify(root: Path = ROOT, base_model_path: Path | None = None) -> dict:
    results = {}
    for study, (prompt_version, step, batch, run_path, record_path) in SPECS.items():
        artifact = root / "models" / f"{study}-abstract"
        manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
        metadata = json.loads((artifact / "roguard_metadata.json").read_text(encoding="utf-8"))
        config = json.loads((artifact / "adapter_config.json").read_text(encoding="utf-8"))
        run = json.loads((root / run_path).read_text(encoding="utf-8"))
        if (manifest.get("artifact_kind") != "experimental_romanian_abstract_card_mlx_adapter" or
                manifest.get("study") != study or
                manifest.get("prompt_version") != prompt_version or
                manifest.get("prompt_sha256") != sha256(root / "src/roguard/prompt.py") or
                manifest.get("adapter_weight_sha256") != sha256(artifact / "adapters.safetensors") or
                manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
                manifest.get("metadata_sha256") != sha256(artifact / "roguard_metadata.json") or
                manifest.get("saved_run_path") != run_path or
                manifest.get("saved_run_sha256") != sha256(root / run_path) or
                manifest.get("comparison_record_path") != record_path or
                manifest.get("comparison_record_sha256") != (
                    sha256(root / record_path) if record_path else None) or
                manifest.get("selected_step") != step or
                metadata.get("base_model_path") is not None or
                metadata.get("base_model_id") != manifest.get("base_model") or
                metadata.get("base_revision") != manifest.get("base_revision") or
                metadata.get("prompt_version") != prompt_version or
                metadata.get("language") != "ro" or
                tuple(metadata.get("categories", ())) != CATEGORIES or
                config != {"fine_tune_type": "lora", "num_layers": 8,
                           "lora_parameters": {"rank": 16, "dropout": 0.05, "scale": 2.0}} or
                run.get("adapter") != f"checkpoints/{study}/adapter" or
                run.get("batches") != [batch] or not run.get("predictions")):
            raise ValueError(f"Historical {study} artifact differs from its frozen record")
        if record_path:
            record = json.loads((root / record_path).read_text(encoding="utf-8"))
            if record.get("adapter_weight_sha256") != manifest["adapter_weight_sha256"]:
                raise ValueError(f"Historical {study} weight differs from frozen comparison")
        reproduced = None
        if base_model_path is not None:
            from roguard import MLXClassifier, screen
            row = load_approved(root, [batch])[0]
            prior = run["predictions"][0]
            backend = MLXClassifier(artifact, base_model_path=base_model_path)
            current = screen(row["text"], language="ro", source_kind=row["source_kind"],
                             backend=backend, thresholds={code: 0.5 for code in CATEGORIES})
            reproduced = prior["id"] == row["id"] and list(current.labels) == prior["predicted"]
            if not reproduced:
                raise ValueError(f"Historical {study} does not reproduce a saved card")
        results[study] = {"weight_sha256": manifest["adapter_weight_sha256"],
                          "saved_label_reproduced": reproduced}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(base_model_path=args.base_model_path), indent=2))
