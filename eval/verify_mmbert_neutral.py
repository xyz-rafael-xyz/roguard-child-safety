"""Verify the frozen benign-request probe without loading a model."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-neutral-0001.json")
DATA = Path("data/probes/ro-neutral-0001.json")
PROTOCOL = Path("docs/NEUTRAL_PROBE_PROTOCOL.md")
RUNNER = Path("eval/run_mmbert_neutral.py")
ARTIFACT = Path("models/ro-mmbert-v11-abstract")


def verify(root: Path = ROOT) -> dict:
    run = json.loads((root / RUN).read_text(encoding="utf-8"))
    manifest = json.loads((root / ARTIFACT / "research.json").read_text(encoding="utf-8"))
    rows = runpy.run_path(str(root / RUNNER))["_rows"](root)
    if (run.get("status") != "frozen_benign_romanian_probe" or
            run.get("data_sha256") != sha256(root / DATA) or
            run.get("protocol_sha256") != sha256(root / PROTOCOL) or
            run.get("runner_sha256") != sha256(root / RUNNER) or
            run.get("adapter_manifest_sha256") != sha256(root / ARTIFACT / "research.json") or
            run.get("adapter_weight_sha256") != manifest["adapter_weight_sha256"] or
            run.get("adapter_weight_sha256") != sha256(root / ARTIFACT / "adapter_model.safetensors") or
            run.get("threshold") != manifest["shared_threshold"]):
        raise ValueError("Frozen benign probe input or adapter differs")
    predictions = run.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise ValueError("Benign probe prediction count differs")
    by_group = {group: 0 for group, _ in rows}
    for index, ((group, _), item) in enumerate(zip(rows, predictions), 1):
        score = item.get("score_D1")
        if (set(item) != {"id", "group", "score_D1", "predicted"} or
                item["id"] != f"neutral-ro-0001-{index:02d}" or
                item["group"] != group or type(score) not in (int, float) or
                not math.isfinite(score) or not 0 <= score <= 1 or
                item["predicted"] != (["D1"] if score >= run["threshold"] else [])):
            raise ValueError(f"Invalid saved benign probe score or decision: {index}")
        by_group[group] += bool(item["predicted"])
    false_reviews = sum(by_group.values())
    success = false_reviews <= 2 and all(count <= 1 for count in by_group.values())
    if (run.get("false_reviews") != false_reviews or
            run.get("false_reviews_by_group") != by_group or
            run.get("preregistered_success") != success):
        raise ValueError("Benign probe stored report differs from decisions")
    return {"requests": len(rows), "false_reviews": false_reviews,
            "by_group": by_group, "max_score_D1": max(item["score_D1"] for item in predictions),
            "preregistered_success": success}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
