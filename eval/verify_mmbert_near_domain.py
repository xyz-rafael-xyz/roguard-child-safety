"""Recompute the frozen near-domain probe from saved scores, without loading models."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.prompt_v18 import FIELDS, decision
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-near-domain-0002.json")


def verify(root: Path = ROOT) -> dict:
    script = runpy.run_path(str(root / "eval/run_mmbert_near_domain.py"))
    data, protocol, runner = script["DATA"], script["PROTOCOL"], script["RUNNER"]
    models, groups = script["MODELS"], script["GROUPS"]
    samples = script["rows"](root)
    record = json.loads((root / RUN).read_text(encoding="utf-8"))
    if (set(record) != {"status", "data_sha256", "protocol_sha256", "runner_sha256",
                        "model_hashes", "results", "never_review_false_reviews",
                        "always_review_false_reviews", "limitation"} or
            record["status"] != "frozen_benign_romanian_near_domain_probe" or
            record["data_sha256"] != sha256(root / data) or
            record["protocol_sha256"] != sha256(root / protocol) or
            record["runner_sha256"] != sha256(root / runner) or
            record["never_review_false_reviews"] != 0 or
            record["always_review_false_reviews"] != len(samples) or
            set(record["model_hashes"]) != set(models) or
            set(record["results"]) != set(models)):
        raise ValueError("Frozen near-domain probe inputs differ")
    for name, directory in models.items():
        manifest = json.loads((root / directory / "research.json").read_text(encoding="utf-8"))
        if record["model_hashes"][name] != {
            "manifest_sha256": sha256(root / directory / "research.json"),
            "weight_sha256": sha256(root / directory / "adapter_model.safetensors"),
        } or manifest["adapter_weight_sha256"] != record["model_hashes"][name]["weight_sha256"]:
            raise ValueError(f"Frozen adapter differs: {name}")
        result = record["results"][name]
        threshold = 0.5 if name != "v11" else 0.9835666418075562
        predictions = result.get("predictions")
        if (result.get("threshold") != threshold or not isinstance(predictions, list) or
                len(predictions) != len(samples)):
            raise ValueError(f"Frozen score count or cutoff differs: {name}")
        for index, ((group, _), item) in enumerate(zip(samples, predictions), 1):
            scores = item.get("scores")
            expected_fields = set(FIELDS) if name == "v18" else {"D1"}
            if (set(item) != {"id", "group", "scores", "predicted"} or
                    item["id"] != f"near-ro-0002-{index:02d}" or
                    item["group"] != group or not isinstance(scores, dict) or
                    set(scores) != expected_fields or
                    any(type(value) not in (int, float) or not math.isfinite(value) or
                        not 0 <= value <= 1 for value in scores.values()) or
                    type(item["predicted"]) is not bool):
                raise ValueError(f"Invalid saved near-domain row: {name} {index}")
            predicted = (decision({field: scores[field] >= threshold for field in FIELDS})
                         if name == "v18" else scores["D1"] >= threshold)
            if item["predicted"] != predicted:
                raise ValueError(f"Saved near-domain decision differs: {name} {index}")
        summary = script["aggregate"](predictions)
        if (set(result) != {"threshold", "predictions", *summary} or any(
                result[key] != value for key, value in summary.items()) or
                set(result["false_reviews_by_group"]) != set(groups)):
            raise ValueError(f"Saved near-domain summary differs: {name}")
    return {"requests": len(samples),
            "false_reviews": {name: record["results"][name]["false_reviews"] for name in models},
            "preregistered_success": {name: record["results"][name]["preregistered_success"]
                                       for name in models}}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
