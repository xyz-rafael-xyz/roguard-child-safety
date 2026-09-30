"""Check private historical mmBERT artifacts against frozen selections and runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "v9a": ("batch-0019", "eval/runs/ro-mmbert-v9a-test-0019.json"),
    "v10": ("batch-0020", "eval/runs/ro-mmbert-v10-test-0020.json"),
}


def verify(root: Path = ROOT, base_model_path: Path | None = None) -> dict:
    results = {}
    for study, (batch, run_name) in SPECS.items():
        artifact = root / "models" / f"ro-mmbert-{study}-abstract"
        manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
        selection_path = root / f"eval/runs/ro-mmbert-{study}-dev-selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        run = json.loads((root / run_name).read_text(encoding="utf-8"))
        config = json.loads((artifact / "adapter_config.json").read_text(encoding="utf-8"))
        if (manifest.get("study") != study or
                manifest.get("adapter_weight_sha256") != sha256(artifact / "adapter_model.safetensors") or
                manifest.get("adapter_weight_sha256") != selection["selected_weight_sha256"] or
                manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
                manifest.get("selection_record_sha256") != sha256(selection_path) or
                manifest.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v4.py") or
                manifest.get("shared_threshold") != selection["selected_threshold"] or
                manifest.get("base_model") != selection["model"] or
                manifest.get("base_revision") != selection["revision"] or
                config.get("base_model_name_or_path") != selection["model"] or
                run.get("batches") != [batch] or
                run.get("selection_record_sha256") != sha256(selection_path) or
                run.get("selected_weight_sha256") != manifest["adapter_weight_sha256"] or
                run.get("threshold") != manifest["shared_threshold"]):
            raise ValueError(f"Historical {study} artifact differs from frozen selection")
        score_delta = None
        if base_model_path is not None:
            from roguard import MMBertResearchClassifier, screen
            rows = load_approved(root, [batch])
            prior = run["predictions"][0]
            backend = MMBertResearchClassifier(base_model_path, artifact)
            current = screen(rows[0]["text"], language="ro",
                             source_kind=rows[0]["source_kind"], backend=backend,
                             thresholds=backend.thresholds)
            code = next(iter(prior["scores"]))
            score_delta = abs(current.scores[code] - prior["scores"][code])
            if (prior["id"] != rows[0]["id"] or score_delta >= 1e-5 or
                    list(current.labels) != prior["predicted"]):
                raise ValueError(f"Historical {study} saved score differs")
        results[study] = {"weight_sha256": manifest["adapter_weight_sha256"],
                          "selected_threshold": manifest["shared_threshold"],
                          "saved_score_delta": score_delta}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(base_model_path=args.base_model_path), indent=2))
