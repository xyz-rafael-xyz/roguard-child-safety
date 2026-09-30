"""Bind prospective author tasks to exact pre-author model and evaluator bytes."""

from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import date
from pathlib import Path

from .review import sha256

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _relative_file(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Study freeze names a path outside the repository")
    target = root / path
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):
        raise ValueError("Study freeze names a missing or symlinked file")
    return target


def _verify_committed_files(root: Path, record_name: str,
                            hashes: dict[str, str], commit: str) -> bool:
    if not (root / ".git").exists():
        raise ValueError("Study freeze needs a Git repository with committed files")
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                      cwd=root, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, check=False).returncode != 0:
        raise ValueError("Study freeze source commit is not an ancestor of HEAD")
    for name in (record_name, *hashes):
        if (subprocess.run(["git", "ls-files", "--error-unmatch", "--", name],
                           cwd=root, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False).returncode != 0 or
                subprocess.run(["git", "diff", "--quiet", "HEAD", "--", name],
                               cwd=root, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False).returncode != 0):
            raise ValueError(f"Study freeze file is not committed at HEAD: {name}")
    if subprocess.run(["git", "diff", "--quiet", commit, "HEAD", "--", *hashes],
                      cwd=root, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, check=False).returncode != 0:
        raise ValueError("Study freeze files lack registered bytes at source commit")
    return True


def verify_study_freeze(root: Path, record_path: Path,
                        language: str, category: str) -> dict:
    """Verify exact bytes and scope, not the human chronology of authorship."""
    root = root.resolve()
    relative = record_path if not record_path.is_absolute() else record_path.relative_to(root)
    if (not relative.parts or relative.parts[:2] != ("eval", "prospective") or
            relative.suffix != ".json"):
        raise ValueError("Study freeze must be under eval/prospective/")
    path = _relative_file(root, str(relative))
    record = json.loads(path.read_text(encoding="utf-8"))
    hashes = record.get("sha256") if isinstance(record, dict) else None
    adapter = record.get("adapter_dir") if isinstance(record, dict) else None
    if (not isinstance(record, dict) or record.get("schema_version") != 1 or
            record.get("status") not in {
                "registered_baseline_before_independent_batch",
                "registered_model_before_independent_batch"} or
            record.get("language") != language or language not in ("ro", "uk") or
            not isinstance(record.get("categories"), list) or
            category not in record["categories"] or
            set(record["categories"]) - {"D1", "S1"} or
            len(record["categories"]) != len(set(record["categories"])) or
            record.get("input_scope") != "independently_authored_abstract_cards_only" or
            not isinstance(record.get("registered_on"), str) or
            not isinstance(record.get("pre_author_git_commit"), str) or
            not _COMMIT.fullmatch(record["pre_author_git_commit"]) or
            not isinstance(adapter, str) or not adapter.startswith("models/") or
            not isinstance(record.get("model_id"), str) or not record["model_id"] or
            type(record.get("cutoff")) not in (int, float) or
            not math.isfinite(record["cutoff"]) or not 0 <= record["cutoff"] <= 1 or
            not isinstance(hashes, dict) or len(hashes) < 4):
        raise ValueError("Study freeze scope or metadata differs")
    try:
        if date.fromisoformat(record["registered_on"]) > date.today():
            raise ValueError("Study freeze date is in the future")
    except ValueError as exc:
        raise ValueError("Study freeze date is invalid") from exc
    manifest_name = f"{adapter}/research.json"
    weights = [name for name in hashes
               if name.startswith(adapter + "/") and name.endswith(".safetensors")]
    code_files = [name for name in hashes if name.startswith("src/roguard/") and name.endswith(".py")]
    if manifest_name not in hashes or len(weights) != 1 or len(code_files) < 2:
        raise ValueError("Study freeze omits the adapter or inference/evaluation code")
    for field in ("predictor_entrypoint", "evaluator_entrypoint"):
        entrypoint = record.get(field)
        if (not isinstance(entrypoint, str) or
                not entrypoint.startswith("src/roguard/") or
                not entrypoint.endswith(".py") or entrypoint not in hashes):
            raise ValueError(f"Study freeze omits the named {field}")
    if record["predictor_entrypoint"] == record["evaluator_entrypoint"]:
        raise ValueError("Study freeze needs distinct prediction and evaluation code")
    for name, digest in hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str) or not _HEX.fullmatch(digest):
            raise ValueError("Study freeze has an invalid file digest")
        if sha256(_relative_file(root, name)) != digest:
            raise ValueError(f"Study freeze file differs: {name}")
    prediction_model_id = record.get("prediction_model_id")
    if (not isinstance(prediction_model_id, str) or
            not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", prediction_model_id)):
        raise ValueError("Study freeze lacks an exact prediction model ID")
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or
            manifest.get("trained_language") != language or
            manifest.get("shared_threshold") != record["cutoff"] or
            manifest.get("adapter_weight_sha256") != hashes[weights[0]]):
        raise ValueError("Study freeze disagrees with the model manifest")
    committed = _verify_committed_files(root, relative.as_posix(), hashes,
                                        record["pre_author_git_commit"])
    return {"path": relative.as_posix(), "sha256": sha256(path),
            "language": language, "category": category,
            "registered_on": record["registered_on"],
            "prediction_model_id": prediction_model_id,
            "registered_files_committed_at_head": committed,
            "human_authorship_chronology_verified": False}
