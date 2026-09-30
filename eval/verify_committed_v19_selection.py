"""Verify the committed V19 adapter and development scores without model loading."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from roguard.mmbert_study import MODEL_ID, REVISION, verify_adapter_head
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path("models/ro-mmbert-v19-joint-abstract")
SELECTION = Path("eval/runs/ro-mmbert-v19-dev-selection.json")


def verify(root: Path = ROOT) -> dict:
    selected = runpy.run_path(str(root / "eval/verify_v19_selection.py"))["verify"](root)
    artifact = root / ARTIFACT
    manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
    config = json.loads((artifact / "adapter_config.json").read_text(encoding="utf-8"))
    epoch = selected["selected_epoch"]
    dev = root / f"eval/runs/ro-mmbert-v19-epoch-{epoch}-dev.json"
    verify_adapter_head(artifact, num_labels=4)
    if (manifest.get("artifact_kind") != "experimental_romanian_abstract_d1_joint_adapter" or
            manifest.get("study") != "v19_joint_factorized_d1" or
            manifest.get("base_model") != MODEL_ID or manifest.get("base_revision") != REVISION or
            manifest.get("adapter_weight_sha256") != sha256(artifact / "adapter_model.safetensors") or
            manifest.get("adapter_weight_sha256") != selected["selected_weight_sha256"] or
            manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
            manifest.get("selection_record_sha256") != sha256(root / SELECTION) or
            manifest.get("development_scores_sha256") != sha256(dev) or
            manifest.get("selected_epoch") != epoch or manifest.get("field_cutoff") != 0.5 or
            manifest.get("development_gate_passed") != selected["development_gate_passed"] or
            manifest.get("evaluation_status") != "selected_development_failed_gate" or
            manifest.get("trained_language") != "ro" or
            manifest.get("trained_domain") != "abstract_symbolic_cards_only" or
            config.get("base_model_name_or_path") != MODEL_ID or
            config.get("r") != 16 or config.get("lora_alpha") != 32):
        raise ValueError("V19 packaged adapter differs from frozen development selection")
    return selected


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
