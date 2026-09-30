"""Package the V16 development-selected adapter before opening its sealed test."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from roguard.mmbert_study import MODEL_ID, REVISION
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
SELECTION = Path("eval/runs/ro-mmbert-v16-dev-selection.json")
DEV = Path("eval/runs/ro-mmbert-v16-epoch-1-dev.json")
TARGET = Path("models/ro-mmbert-v16-abstract")


def package(root: Path = ROOT) -> Path:
    selection = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    source_selection = root / "checkpoints/ro-mmbert-v16/development-selection.json"
    if (sha256(root / SELECTION) != sha256(source_selection) or
            selection.get("study") != "v16_advisory_prose_repair" or
            selection.get("model") != MODEL_ID or selection.get("revision") != REVISION or
            selection.get("source_adapter_weight_sha256") != sha256(
                root / "models/ro-mmbert-v11-abstract/adapter_model.safetensors") or
            selection.get("protocol_sha256") != sha256(root / "docs/V16_PROTOCOL.md") or
            selection.get("prefit_amendment_sha256") != sha256(root / "docs/V16_PREFIT_AMENDMENT.md") or
            selection.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v4.py") or
            selection.get("selected_epoch") != 1 or
            selection.get("selected_threshold") != 0.5):
        raise ValueError("V16 development selection differs from registered inputs")
    best = max(selection["candidates"], key=lambda c: (
        c["report"]["exact_pairs"], c["report"]["exact_cards"],
        -c["report"]["false_review_on_negatives"],
        -abs(c["threshold"] - 0.5), -c["epoch"], -c["threshold"]))
    source = root / "checkpoints/ro-mmbert-v16/epoch-1"
    source_weight = source / "adapter_model.safetensors"
    source_dev = root / "checkpoints/ro-mmbert-v16/epoch-1-dev.json"
    if (best["epoch"] != selection["selected_epoch"] or
            best["weight_sha256"] != selection["selected_weight_sha256"] or
            best["weight_sha256"] != sha256(source_weight) or
            best["dev_predictions_sha256"] != sha256(source_dev) or
            sha256(source_dev) != sha256(root / DEV)):
        raise ValueError("V16 selected weight or development scores differ")
    target = root / TARGET
    if target.exists():
        raise FileExistsError("Preserve the first packaged V16 adapter")
    target.mkdir(parents=True)
    shutil.copyfile(source_weight, target / "adapter_model.safetensors")
    config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = MODEL_ID
    (target / "adapter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "artifact_kind": "experimental_romanian_abstract_card_adapter",
        "study": "v16_advisory_prose_repair", "base_model": MODEL_ID,
        "base_revision": REVISION,
        "source_adapter_weight_sha256": selection["source_adapter_weight_sha256"],
        "adapter_weight_sha256": sha256(target / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(target / "adapter_config.json"),
        "selection_record_sha256": sha256(root / SELECTION),
        "development_scores_sha256": sha256(root / DEV),
        "prompt_sha256": selection["prompt_sha256"],
        "shared_threshold": selection["selected_threshold"],
        "evaluation_status": "selected_development_only",
        "trained_language": "ro", "trained_domain": "abstract_symbolic_cards_only",
    }
    (target / "research.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        "# V16 Romanian abstract advisory adapter\n\n"
        "Private research artifact selected on development batch 0029. Its separate "
        "test result is recorded in BENCHMARK.md. It is not a detector for real child "
        "messages or a calibrated risk model. Load it only with the pinned mmBERT base.\n",
        encoding="utf-8",
    )
    return target


if __name__ == "__main__":
    print(package())
