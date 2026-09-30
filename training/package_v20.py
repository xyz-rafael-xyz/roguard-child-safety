"""Package the development-selected V20 adapter with its failed-gate record."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

from roguard.mmbert_study import MODEL_ID, REVISION, verify_adapter_head
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
SELECTION = Path("eval/runs/ro-mmbert-v20-dev-selection.json")
TARGET = Path("models/ro-mmbert-v20-balanced-joint-abstract")


def package(root: Path = ROOT) -> Path:
    checked = runpy.run_path(str(root / "eval/verify_v20_selection.py"))["verify"](root)
    epoch = checked["selected_epoch"]
    source = root / f"checkpoints/ro-mmbert-v20/epoch-{epoch}"
    source_selection = root / "checkpoints/ro-mmbert-v20/development-selection.json"
    source_dev = root / f"checkpoints/ro-mmbert-v20/epoch-{epoch}-dev.json"
    committed_dev = root / f"eval/runs/ro-mmbert-v20-epoch-{epoch}-dev.json"
    if (sha256(source_selection) != sha256(root / SELECTION) or
            sha256(source_dev) != sha256(committed_dev) or
            sha256(source / "adapter_model.safetensors") != checked["selected_weight_sha256"]):
        raise ValueError("V20 selected checkpoint or scores differ")
    target = root / TARGET
    if target.exists():
        raise FileExistsError("Preserve prior V20 packaged adapter")
    target.mkdir(parents=True)
    shutil.copyfile(source / "adapter_model.safetensors", target / "adapter_model.safetensors")
    config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = MODEL_ID
    (target / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verify_adapter_head(target, num_labels=4)
    manifest = {
        "artifact_kind": "experimental_romanian_abstract_d1_balanced_joint_adapter",
        "study": "v20_balanced_joint_d1", "base_model": MODEL_ID,
        "base_revision": REVISION,
        "adapter_weight_sha256": sha256(target / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(target / "adapter_config.json"),
        "selection_record_sha256": sha256(root / SELECTION),
        "development_scores_sha256": sha256(committed_dev),
        "selected_epoch": epoch, "field_cutoff": 0.5,
        "development_gate_passed": checked["development_gate_passed"],
        "evaluation_status": "selected_development_failed_gate",
        "trained_language": "ro", "trained_domain": "abstract_symbolic_cards_only",
    }
    (target / "research.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        "# V20 Romanian abstract D1 balanced joint-field adapter\n\n"
        "Private research artifact selected despite a failed development gate. "
        "It uses a class-balanced four-field head and fixed Boolean decision rule "
        "for invented state cards. It is not a detector for child messages or a "
        "calibrated risk model. Load only with the pinned mmBERT base.\n",
        encoding="utf-8")
    return target


if __name__ == "__main__":
    print(package())
