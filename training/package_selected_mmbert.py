"""Package hash-selected historical mmBERT research adapters for private Git."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from roguard.mmbert_study import MODEL_ID, REVISION
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "v9a": (Path("checkpoints/ro-mmbert-v9a/epoch-5"),
            Path("eval/runs/ro-mmbert-v9a-dev-selection.json")),
    "v10": (Path("checkpoints/ro-mmbert-v10/epoch-2"),
            Path("eval/runs/ro-mmbert-v10-dev-selection.json")),
}


def package(study: str, root: Path = ROOT) -> Path:
    if study not in SOURCES:
        raise ValueError("Unknown historical selected adapter")
    source, selection_path = SOURCES[study]
    source, selection_path = root / source, root / selection_path
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_weight = source / "adapter_model.safetensors"
    if (selection["model"] != MODEL_ID or selection["revision"] != REVISION or
            selection["prompt_sha256"] != sha256(root / "src/roguard/prompt_v4.py") or
            selection["selected_weight_sha256"] != sha256(source_weight) or
            source.name != f"epoch-{selection['selected_epoch']}"):
        raise ValueError("Historical selected weight or prompt differs")
    target = root / "models" / f"ro-mmbert-{study}-abstract"
    if target.exists():
        raise FileExistsError("Preserve an existing packaged adapter")
    target.mkdir(parents=True)
    shutil.copyfile(source_weight, target / "adapter_model.safetensors")
    config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = MODEL_ID
    (target / "adapter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "artifact_kind": "experimental_romanian_abstract_card_adapter",
        "study": study, "base_model": MODEL_ID, "base_revision": REVISION,
        "adapter_weight_sha256": sha256(target / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(target / "adapter_config.json"),
        "selection_record_sha256": sha256(selection_path),
        "prompt_sha256": selection["prompt_sha256"],
        "shared_threshold": selection["selected_threshold"],
        "evaluation_status": "failed_registered_synthetic_target",
        "trained_language": "ro", "trained_domain": "abstract_symbolic_cards_only",
    }
    (target / "research.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=tuple(SOURCES))
    args = parser.parse_args()
    print(package(args.study))


if __name__ == "__main__":
    main()
