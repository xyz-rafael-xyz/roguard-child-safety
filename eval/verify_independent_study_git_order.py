"""Check the committed order of a sealed blind prediction and its labeled batch."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from roguard.review import sha256
from roguard.study_freeze import verify_study_freeze

ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = re.compile(r"batch-[0-9]{4,8}\Z")
PREDICTION_FIELDS = {"schema_version", "packet_sha256", "language", "category",
                     "model_id", "predictions"}
HEX = re.compile(r"[0-9a-f]{64}\Z")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    if result.returncode:
        raise ValueError(f"Git history check failed: {' '.join(args[:2])}")
    return result.stdout.strip()


def _relative_file(root: Path, path: Path, parent: str) -> str:
    target = path if path.is_absolute() else root / path
    if target.is_symlink() or not target.is_file():
        raise ValueError("Study chronology needs an ordinary existing file")
    resolved = target.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Study chronology file is outside the repository")
    relative = resolved.relative_to(root).as_posix()
    if not relative.startswith(parent + "/") or not relative.endswith(".json"):
        raise ValueError(f"Study chronology needs a JSON file under {parent}/")
    return relative


def _immutable_addition(root: Path, relative: str) -> str:
    if _git(root, "status", "--porcelain", "--", relative):
        raise ValueError(f"Study chronology file differs from committed HEAD: {relative}")
    changes = _git(root, "log", "--format=%H", "--", relative).splitlines()
    additions = _git(root, "log", "--diff-filter=A", "--format=%H", "--", relative).splitlines()
    if len(changes) != 1 or additions != changes:
        raise ValueError(f"Study chronology file must have one immutable addition: {relative}")
    return changes[0]


def _strict_ancestor(root: Path, earlier: str, later: str, description: str) -> None:
    if earlier == later or subprocess.run(
            ["git", "merge-base", "--is-ancestor", earlier, later], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False).returncode != 0:
        raise ValueError(f"Study chronology requires {description}")


def verify(root: Path, batch: str, predictions_path: Path) -> dict:
    root = root.resolve()
    if not BATCH_ID.fullmatch(batch):
        raise ValueError("Study chronology needs a batch ID")
    if _git(root, "rev-parse", "--is-shallow-repository") != "false":
        raise ValueError("Study chronology needs complete Git history")
    names = [f"data/synthetic/{batch}{suffix}" for suffix in
             (".jsonl", ".preview.md", ".provenance.json", ".review.json")]
    prediction_name = _relative_file(root, predictions_path, "eval/prospective")
    prediction = json.loads((root / prediction_name).read_text(encoding="utf-8"))
    provenance = json.loads((root / names[2]).read_text(encoding="utf-8"))
    if (not isinstance(prediction, dict) or set(prediction) != PREDICTION_FIELDS or
            prediction.get("schema_version") != 2 or
            not isinstance(prediction.get("packet_sha256"), str) or
            not HEX.fullmatch(prediction["packet_sha256"]) or
            not isinstance(prediction.get("predictions"), list) or
            len(prediction["predictions"]) != 96 or
            not isinstance(provenance, dict) or provenance.get("batch") != batch or
            provenance.get("label_status") != "author_intended_not_adjudicated" or
            prediction.get("language") != provenance.get("language") or
            prediction.get("category") != provenance.get("category") or
            not isinstance(provenance.get("model_freeze_path"), str) or
            not isinstance(provenance.get("model_freeze_sha256"), str)):
        raise ValueError("Study chronology prediction or author provenance differs")
    freeze = verify_study_freeze(root, Path(provenance["model_freeze_path"]),
                                 provenance["language"], provenance["category"])
    if (freeze["sha256"] != provenance["model_freeze_sha256"] or
            prediction.get("model_id") != freeze["prediction_model_id"]):
        raise ValueError("Study chronology prediction differs from the registered freeze")
    freeze_commit = _immutable_addition(root, freeze["path"])
    prediction_commit = _immutable_addition(root, prediction_name)
    batch_commits = {_immutable_addition(root, name) for name in names}
    if len(batch_commits) != 1:
        raise ValueError("Study chronology needs all approved batch files in one commit")
    batch_commit = next(iter(batch_commits))
    _strict_ancestor(root, freeze_commit, prediction_commit,
                     "the freeze commit before blind predictions")
    _strict_ancestor(root, prediction_commit, batch_commit,
                     "the blind prediction commit before labeled batch files")
    return {
        "batch": batch, "predictions_sha256": sha256(root / prediction_name),
        "model_freeze_sha256": freeze["sha256"],
        "freeze_commit": freeze_commit, "prediction_commit": prediction_commit,
        "labeled_batch_commit": batch_commit,
        "git_commit_order_verified": True,
        "external_push_order_verified": False,
        "model_operator_blinding_verified": False,
        "human_authorship_chronology_verified": False,
        "adjudication_validity_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root, args.batch, args.predictions), indent=2))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
