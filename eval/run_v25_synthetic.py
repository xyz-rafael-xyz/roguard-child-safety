"""Score the once-revealed V25 same-origin synthetic test without reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from eval.v25_metrics import CELLS, evaluate, target_passed
from roguard.review import sha256, taxonomy_sha256
from roguard.v25_abstract import V25SyntheticResearchClassifier
from training.generate_v25_synthetic import (COMMITMENT, DATA, PRIVATE_SEED,
                                             PUBLIC_SEEDS, ROOT, build_rows)
from training.train_v25_synthetic import char_baseline_model, char_baseline_scores

ARTIFACT = Path("models/bi-mmbert-v25-abstract")
RESULT = Path("eval/prospective/v25-synthetic-result.json")
PREDICTIONS = Path("eval/prospective/v25-synthetic-predictions.jsonl")


def _rows(root: Path, split: str) -> list[dict]:
    return [json.loads(line) for line in (root / DATA / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines()]


def _bytes_json(data: dict) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _bytes_jsonl(data: dict) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def verify_frozen_study(root: Path) -> tuple[dict, dict, dict, list[dict], list[dict], list[dict]]:
    """Check the committed pretest inputs and selected artifact before any scoring."""
    root = root.resolve()
    commitment = json.loads((root / COMMITMENT).read_text(encoding="utf-8"))
    artifact = json.loads((root / ARTIFACT / "research.json").read_text(encoding="utf-8"))
    selection_path = root / ARTIFACT / "development-selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (commitment.get("status") != "sealed_same_origin_synthetic_test_not_yet_generated" or
            commitment.get("test_rows_at_registration") != 0 or
            artifact.get("status") != "selected_before_test_reveal" or
            artifact.get("study") != "v25_bilingual_synthetic" or
            artifact.get("test_seed_sha256") != commitment.get("test_seed_sha256") or
            artifact.get("test_seed_commitment_sha256") != sha256(root / COMMITMENT) or
            artifact.get("development_selection_sha256") != sha256(selection_path) or
            artifact.get("selected_epoch") != selection.get("selected_epoch") or
            artifact.get("thresholds") != selection.get("selected_thresholds") or
            artifact.get("weight_sha256") != selection.get("selected_weight_sha256") or
            artifact.get("prompt_sha256") != sha256(root / "src/roguard/v25_abstract.py") or
            artifact.get("trainer_sha256") != sha256(root / "training/train_v25_synthetic.py") or
            artifact.get("metrics_sha256") != sha256(root / "eval/v25_metrics.py") or
            artifact.get("protocol_sha256") != sha256(root / "docs/V25_SYNTHETIC_PROTOCOL.md") or
            commitment.get("generator_sha256") != sha256(root / "training/generate_v25_synthetic.py") or
            commitment.get("romanian_wording_sha256") != sha256(root / "training/v25_ro.py") or
            commitment.get("ukrainian_wording_sha256") != sha256(root / "training/v25_uk.py") or
            commitment.get("taxonomy_sha256") != {
                language: taxonomy_sha256(root, language) for language in ("ro", "uk")
            }):
        raise ValueError("V25 prospective commitment, selected model, or source differs")
    for relative in (COMMITMENT, ARTIFACT / "research.json",
                     ARTIFACT / "development-selection.json",
                     ARTIFACT / "adapter_model.safetensors"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(relative)],
                          cwd=root, capture_output=True, check=False).returncode:
            raise ValueError(f"V25 frozen selection is not committed: {relative}")
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(relative)],
                          cwd=root, capture_output=True, check=False).returncode:
            raise ValueError(f"V25 selected artifact differs from its committed version: {relative}")
    seed = (root / PRIVATE_SEED).read_text(encoding="ascii").strip()
    if hashlib.sha256(seed.encode("ascii")).hexdigest() != commitment["test_seed_sha256"]:
        raise ValueError("V25 revealed seed differs from its prefit commitment")
    train, dev, test = (_rows(root, split) for split in ("train", "dev", "test"))
    for split, rows in (("train", train), ("dev", dev)):
        if (sha256(root / DATA / f"{split}.jsonl") != commitment[f"{split}_sha256"] or
                rows != build_rows(split, PUBLIC_SEEDS[split])):
            raise ValueError(f"V25 {split} is not the committed generator output")
    if test != build_rows("test", seed):
        raise ValueError("V25 test does not match the sealed seed and generator")
    if (len(test) != 384 or len(train) != 1536 or len(dev) != 384 or
            set(artifact["thresholds"]) != set(CELLS)):
        raise ValueError("V25 split size or selected cutoffs differ")
    return commitment, artifact, selection, train, dev, test


def run(base: Path, root: Path = ROOT) -> dict:
    root = root.resolve()
    if (root / RESULT).exists() or (root / PREDICTIONS).exists():
        raise FileExistsError("Preserve the existing V25 test report and predictions")
    commitment, artifact, selection, train, dev, test = verify_frozen_study(root)
    classifier = V25SyntheticResearchClassifier(base, root / ARTIFACT)
    model_scores = classifier.score_cards(test)
    model_report = evaluate(test, model_scores, artifact["thresholds"])
    baseline_model = char_baseline_model(train)
    baseline_dev_scores = char_baseline_scores(baseline_model, dev)
    stored_baseline = selection["char_baseline_dev"]
    if (stored_baseline["scores"] != baseline_dev_scores or
            stored_baseline["report"] != evaluate(dev, baseline_dev_scores,
                                                    stored_baseline["thresholds"])):
        raise ValueError("V25 lexical comparator is not reproducible from its frozen training rows")
    baseline_scores = char_baseline_scores(baseline_model, test)
    baseline_report = evaluate(test, baseline_scores, stored_baseline["thresholds"])
    never = evaluate(test, [0.0] * len(test), {cell: 0.5 for cell in CELLS})
    always = evaluate(test, [1.0] * len(test), {cell: 0.5 for cell in CELLS})
    predictions = b"".join(_bytes_jsonl({
        "id": row["id"], "expected": row["labels"],
        "model_score": model_score,
        "model_predicted": [row["labels"][0] if row["labels"] else
                            ("D1" if row["source_kind"] == "message" else "S1")]
                           if model_score >= artifact["thresholds"][
                               f"{row['language']}:{'D1' if row['source_kind'] == 'message' else 'S1'}"]
                           else [],
        "char_baseline_score": baseline_score,
    }) for row, model_score, baseline_score in zip(test, model_scores, baseline_scores))
    result = {
        "study": "v25_bilingual_synthetic", "scope": "invented_abstract_metadata_only",
        "independent_human_authorship": False, "independent_language_review": False,
        "eligible_for_live_child_message_screening": False,
        "test_seed_sha256": commitment["test_seed_sha256"],
        "test_sha256": sha256(root / DATA / "test.jsonl"),
        "selected_model_manifest_sha256": sha256(root / ARTIFACT / "research.json"),
        "selected_weight_sha256": artifact["weight_sha256"],
        "predictions_sha256": hashlib.sha256(predictions).hexdigest(),
        "model": model_report,
        "fixed_target_passed": target_passed(model_report),
        "char_ngram_baseline": baseline_report,
        "always_review": always, "never_review": never,
    }
    (root / PREDICTIONS).parent.mkdir(parents=True, exist_ok=True)
    with (root / PREDICTIONS).open("xb") as stream:
        stream.write(predictions)
    with (root / RESULT).open("xb") as stream:
        stream.write(_bytes_json(result))
    return {"result": str(RESULT), "test_pairs": model_report["pairs"],
            "exact_pairs": model_report["exact_pairs"],
            "fixed_target_passed": result["fixed_target_passed"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.base_model_path), indent=2))
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
