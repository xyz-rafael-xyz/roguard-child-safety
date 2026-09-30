"""Package the development-selected V18 factor adapter for a frozen diagnostic test."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

from roguard.mmbert_study import MODEL_ID, REVISION
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
SELECTION = Path("eval/runs/ro-mmbert-v18-dev-selection.json")
TARGET = Path("models/ro-mmbert-v18-factor-abstract")


def package(root: Path = ROOT) -> Path:
    checked = runpy.run_path(str(root / "eval/verify_v18_selection.py"))["verify"](root)
    selection = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    epoch = checked["selected_epoch"]
    source = root / f"checkpoints/ro-mmbert-v18/epoch-{epoch}"
    source_weight = source / "adapter_model.safetensors"
    source_selection = root / "checkpoints/ro-mmbert-v18/development-selection.json"
    source_dev = root / f"checkpoints/ro-mmbert-v18/epoch-{epoch}-dev.json"
    committed_dev = root / f"eval/runs/ro-mmbert-v18-epoch-{epoch}-dev.json"
    if (sha256(source_selection) != sha256(root / SELECTION) or
            sha256(source_weight) != checked["selected_weight_sha256"] or
            sha256(source_dev) != sha256(committed_dev)):
        raise ValueError("V18 source checkpoint or scores differ from selection")
    target = root / TARGET
    if target.exists():
        raise FileExistsError("Preserve previously packaged V18 adapter")
    target.mkdir(parents=True)
    shutil.copyfile(source_weight, target / "adapter_model.safetensors")
    config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = MODEL_ID
    (target / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "artifact_kind": "experimental_romanian_abstract_d1_factor_adapter",
        "study": "v18_factorized_d1", "base_model": MODEL_ID,
        "base_revision": REVISION,
        "source_adapter_weight_sha256": selection["source_adapter_weight_sha256"],
        "adapter_weight_sha256": sha256(target / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(target / "adapter_config.json"),
        "selection_record_sha256": sha256(root / SELECTION),
        "development_scores_sha256": sha256(committed_dev),
        "prompt_sha256": selection["prompt_sha256"],
        "selected_epoch": epoch, "factor_cutoff": 0.5,
        "development_gate_passed": checked["development_gate_passed"],
        "evaluation_status": "selected_development_failed_gate",
        "trained_language": "ro", "trained_domain": "abstract_symbolic_cards_only",
    }
    (target / "research.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        "# V18 Romanian abstract D1 factor adapter\n\n"
        "Private research artifact selected on development batch 0033 despite a failed "
        "development gate. It predicts four abstract state fields and applies a fixed Boolean "
        "rule. See BENCHMARK.md for the subsequent diagnostic test. It is not a detector "
        "for child messages or a calibrated risk model. Load only with the pinned mmBERT base.\n",
        encoding="utf-8",
    )
    return target


if __name__ == "__main__":
    print(package())
