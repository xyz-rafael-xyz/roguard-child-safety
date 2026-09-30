"""Verify the pre-author V16 independent-study freeze without reading cards."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from roguard.study_freeze import verify_study_freeze

ROOT = Path(__file__).resolve().parents[1]
RECORD = Path("eval/prospective/ro-v16-registered-study-freeze-v5.json")


def _source_files_at_commit(root: Path, commit: str) -> set[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit, "--", "src/roguard"],
        cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise ValueError("Registered study freeze source commit is unavailable")
    return {name for name in result.stdout.splitlines() if name.endswith(".py")}


def verify(root: Path = ROOT) -> dict:
    record = json.loads((root / RECORD).read_text(encoding="utf-8"))
    if record.get("supersedes") != "eval/prospective/ro-v16-registered-study-freeze-v4.json":
        raise ValueError("Registered study freeze does not identify its superseded draft")
    source_files = _source_files_at_commit(root, record["pre_author_git_commit"])
    if (not source_files or not source_files.issubset(record["sha256"]) or
            "docs/STUDY_CHRONOLOGY_ADDENDUM.md" not in record["sha256"] or
            "docs/STUDY_PREDICTION_SEAL.md" not in record["sha256"] or
            "eval/verify_independent_study_git_order.py" not in record["sha256"]):
        raise ValueError("Registered study freeze omits pre-author RoGuard source code")
    if (record.get("predictor_entrypoint") != "src/roguard/prospective_predict.py" or
            record.get("evaluator_entrypoint") != "src/roguard/human_eval_registered.py" or
            record.get("prediction_model_id") !=
            "mmbert_" + record["sha256"]["models/ro-mmbert-v16-abstract/adapter_model.safetensors"]):
        raise ValueError("Registered study freeze names a different V16 prediction path")
    outcomes = [verify_study_freeze(root, RECORD, "ro", category)
                for category in ("D1", "S1")]
    if not all(item["registered_files_committed_at_head"] for item in outcomes):
        raise ValueError("Registered study freeze is not committed")
    return {"status": "registered_before_independent_batch",
            "record_sha256": outcomes[0]["sha256"],
            "files_verified": len(record["sha256"]),
            "categories": [item["category"] for item in outcomes],
            "human_authorship_chronology_verified": False,
            "real_child_language_accuracy_established": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
