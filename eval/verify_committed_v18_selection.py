"""Verify selected V18 adapter and development scores without loading a base model."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from roguard.mmbert_study import MODEL_ID, REVISION
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path("models/ro-mmbert-v18-factor-abstract")
SELECTION = Path("eval/runs/ro-mmbert-v18-dev-selection.json")


def verify(root: Path = ROOT) -> dict:
    checked = runpy.run_path(str(root / "eval/verify_v18_selection.py"))["verify"](root)
    selection = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    artifact = root / ARTIFACT
    manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
    config = json.loads((artifact / "adapter_config.json").read_text(encoding="utf-8"))
    epoch = checked["selected_epoch"]
    dev = root / f"eval/runs/ro-mmbert-v18-epoch-{epoch}-dev.json"
    if (manifest.get("artifact_kind") != "experimental_romanian_abstract_d1_factor_adapter" or
            manifest.get("study") != "v18_factorized_d1" or
            manifest.get("base_model") != MODEL_ID or manifest.get("base_revision") != REVISION or
            manifest.get("source_adapter_weight_sha256") != selection["source_adapter_weight_sha256"] or
            manifest.get("adapter_weight_sha256") != sha256(artifact / "adapter_model.safetensors") or
            manifest.get("adapter_weight_sha256") != checked["selected_weight_sha256"] or
            manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
            manifest.get("selection_record_sha256") != sha256(root / SELECTION) or
            manifest.get("development_scores_sha256") != sha256(dev) or
            manifest.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v18.py") or
            manifest.get("selected_epoch") != epoch or
            manifest.get("factor_cutoff") != 0.5 or
            manifest.get("development_gate_passed") != checked["development_gate_passed"] or
            manifest.get("evaluation_status") != "selected_development_failed_gate" or
            config.get("base_model_name_or_path") != MODEL_ID or
            config.get("r") != 16 or config.get("lora_alpha") != 32):
        raise ValueError("V18 packaged adapter differs from selected development run")
    return checked


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
