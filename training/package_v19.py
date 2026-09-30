"""Package the selected V19 checkpoint after verifying frozen development scores."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

from roguard.mmbert_study import MODEL_ID, REVISION, verify_adapter_head
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
SELECTION = Path("eval/runs/ro-mmbert-v19-dev-selection.json")
TARGET = Path("models/ro-mmbert-v19-joint-abstract")


def package(root: Path = ROOT) -> Path:
    checked = runpy.run_path(str(root / "eval/verify_v19_selection.py"))["verify"](root)
    epoch = checked["selected_epoch"]
    source = root / f"checkpoints/ro-mmbert-v19/epoch-{epoch}"
    original_selection = root / "checkpoints/ro-mmbert-v19/development-selection.json"
    original_dev = root / f"checkpoints/ro-mmbert-v19/epoch-{epoch}-dev.json"
    committed_dev = root / f"eval/runs/ro-mmbert-v19-epoch-{epoch}-dev.json"
    if (sha256(original_selection) != sha256(root / SELECTION) or
            sha256(original_dev) != sha256(committed_dev) or
            sha256(source / "adapter_model.safetensors") != checked["selected_weight_sha256"]):
        raise ValueError("V19 checkpoint or scores differ from frozen development selection")
    target = root / TARGET
    if target.exists():
        raise FileExistsError("Preserve previously packaged V19 adapter")
    target.mkdir(parents=True)
    shutil.copyfile(source / "adapter_model.safetensors", target / "adapter_model.safetensors")
    config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = MODEL_ID
    (target / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verify_adapter_head(target, num_labels=4)
    manifest = {
        "artifact_kind": "experimental_romanian_abstract_d1_joint_adapter",
        "study": "v19_joint_factorized_d1", "base_model": MODEL_ID,
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
        "# V19 Romanian abstract D1 joint-field adapter\n\n"
        "Private research artifact selected after a failed development gate. "
        "It scores four invented state fields jointly and uses a fixed Boolean decision rule. "
        "It is not a detector for child messages or a calibrated risk model. "
        "Load only with the pinned mmBERT base; see BENCHMARK.md for the diagnostic test.\n",
        encoding="utf-8")
    return target


if __name__ == "__main__":
    print(package())
