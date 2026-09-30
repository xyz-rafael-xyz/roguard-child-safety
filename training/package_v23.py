"""Package the V23 development-selected failure without changing its cutoff."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from roguard.mmbert_study import MODEL_ID, REVISION, verify_adapter_head
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("checkpoints/ro-mmbert-v23")
SELECTION = Path("eval/runs/ro-mmbert-v23-dev-selection.json")
TARGET = Path("models/ro-mmbert-v23-abstract")


def package(root: Path = ROOT) -> Path:
    source_record = root / SOURCE / "development-selection.json"
    record = json.loads(source_record.read_text(encoding="utf-8"))
    if (record.get("study") != "v23_group_dro_worst_surface_d1" or
            record.get("model") != MODEL_ID or record.get("revision") != REVISION or
            record.get("source_adapter_weight_sha256") != sha256(
                root / "models/ro-mmbert-v16-abstract/adapter_model.safetensors") or
            record.get("protocol_sha256") != sha256(root / "docs/V23_PROTOCOL.md") or
            record.get("trainer_sha256") != sha256(root / "training/train_mmbert_v23.py") or
            record.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v4.py")):
        raise ValueError("V23 selected source differs from registration")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or [item.get("epoch") for item in candidates] != [1, 2, 3]:
        raise ValueError("V23 epoch records are incomplete")
    best = max(candidates, key=lambda item: (
        item["summary"]["worst_surface_pairs"], item["summary"]["total_pairs"],
        -item["summary"]["false_reviews"], -item["epoch"]))
    if (record.get("selected_epoch") != best["epoch"] or
            record.get("selected_weight_sha256") != best["weight_sha256"] or
            record.get("development_gate_passed") is not False):
        raise ValueError("V23 failed development selection differs")
    for candidate in candidates:
        epoch = candidate["epoch"]
        checkpoint = root / SOURCE / f"epoch-{epoch}"
        raw = root / SOURCE / f"epoch-{epoch}-dev.json"
        if (sha256(checkpoint / "adapter_model.safetensors") != candidate["weight_sha256"] or
                sha256(checkpoint / "adapter_config.json") != candidate["adapter_config_sha256"] or
                sha256(raw) != candidate["dev_predictions_sha256"]):
            raise ValueError(f"V23 epoch {epoch} artifacts differ")
    destinations = [root / SELECTION]
    destinations.extend(root / f"eval/runs/ro-mmbert-v23-epoch-{epoch}-dev.json"
                        for epoch in (1, 2, 3))
    target = root / TARGET
    if target.exists() or any(path.exists() for path in destinations):
        raise FileExistsError("Preserve existing V23 package or score files")
    shutil.copyfile(source_record, destinations[0])
    for epoch, path in enumerate(destinations[1:], 1):
        shutil.copyfile(root / SOURCE / f"epoch-{epoch}-dev.json", path)
    source_adapter = root / SOURCE / f"epoch-{best['epoch']}"
    verify_adapter_head(source_adapter)
    target.mkdir(parents=True)
    shutil.copyfile(source_adapter / "adapter_model.safetensors",
                    target / "adapter_model.safetensors")
    config = json.loads((source_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = MODEL_ID
    (target / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "artifact_kind": "experimental_romanian_abstract_card_adapter",
        "study": "v23_group_dro_worst_surface_d1", "base_model": MODEL_ID,
        "base_revision": REVISION,
        "source_adapter_weight_sha256": record["source_adapter_weight_sha256"],
        "adapter_weight_sha256": sha256(target / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(target / "adapter_config.json"),
        "selection_record_sha256": sha256(root / SELECTION),
        "development_scores_sha256": sha256(destinations[best["epoch"]]),
        "selected_epoch": best["epoch"], "shared_threshold": 0.5,
        "development_gate_passed": False,
        "evaluation_status": "selected_development_only",
        "prompt_sha256": record["prompt_sha256"], "trained_language": "ro",
        "trained_domain": "abstract_symbolic_cards_only",
    }
    (target / "research.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        "# V23 Romanian abstract D1 adapter\n\n"
        "Private research candidate selected on consumed development batches 0037 and 0038. "
        "Its registered development gate failed. It has no new held-out test and must not "
        "screen child messages. Load only with the pinned mmBERT base.\n",
        encoding="utf-8")
    return target


if __name__ == "__main__":
    print(package())
