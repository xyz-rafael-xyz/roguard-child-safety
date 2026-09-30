"""Prospective V28 bilingual metadata cards for a pinned hybrid reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import secrets
import subprocess
from pathlib import Path

from roguard.review import sha256, taxonomy_sha256
from training.generate_v25_synthetic import (_d1_oracle, _d1_pair, _rng,
                                             _s1_oracle, _s1_pair, _write_new)
from training import v28_ro, v28_uk

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/synthetic/v28")
COMMITMENT = Path("eval/prospective/v28-hybrid-seed-commitment.json")
PRIVATE_SEED = Path("review_runs/v28-hybrid-test-seed.txt")
ARTIFACT = Path("models/bi-v28-hybrid")
PUBLIC_SEEDS = {"train": "v28-hybrid-train-2026-10-01",
                "dev": "v28-hybrid-dev-2026-10-01"}
COUNTS = {"train": 192, "dev": 48, "test": 48}
ORIGIN = "v28_same_origin_hybrid_metadata_v1"
LANGUAGE = {"ro": v28_ro, "uk": v28_uk}


def _style(split: str, index: int, factor_count: int) -> int:
    if split == "train":
        return (index // factor_count) % 9
    return 9 if split == "dev" else 10


def _render(language: str, code: str, facts, style_index: int,
            shuffle_seed: int, marker: str) -> str:
    module = LANGUAGE[language]
    style = (module.D1_STYLES if code == "D1" else module.S1_STYLES)[style_index]
    if code == "D1":
        clauses = [style["role"][facts.source_role != "minor"],
                   style["anchor"][not facts.safety_or_support_anchor],
                   style["indirect"][not facts.indirect_or_repeated_support_pattern],
                   style["explicit"][not facts.explicit_support_request]]
    else:
        clauses = [style["field"].format(module.FIELDS[facts.field]),
                   style["applicable"][not facts.applicable],
                   style["passed"][not facts.passed]]
    random.Random(shuffle_seed).shuffle(clauses)
    return f"{style['open']} {marker}: " + "; ".join(clauses) + f". {style['close']}"


def build_rows(split: str, seed: str) -> list[dict]:
    if split not in COUNTS or not isinstance(seed, str) or len(seed) < 16:
        raise ValueError("V28 requires a registered split and nontrivial seed")
    rows = []
    for language in ("ro", "uk"):
        for code in ("D1", "S1"):
            for index in range(COUNTS[split]):
                rng = _rng(seed, language, code, index)
                factor, negative, positive = (_d1_pair(index, rng) if code == "D1"
                                              else _s1_pair(index))
                style = _style(split, index, 4 if code == "D1" else 2)
                marker = hashlib.sha256(
                    f"v28-marker:{seed}:{language}:{code}:{index}".encode()
                ).hexdigest()[:10]
                shuffle_seed = rng.randrange(1 << 60)
                for variant, facts in enumerate((negative, positive)):
                    if bool(variant) != (_d1_oracle(facts) if code == "D1" else _s1_oracle(facts)):
                        raise AssertionError("V28 latent facts and policy oracle differ")
                    rows.append({
                        "id": f"v28-{split}-{language}-{code.lower()}-{index:04d}-{variant}",
                        "language": language,
                        "source_kind": "message" if code == "D1" else "response",
                        "split": split,
                        "text": _render(language, code, facts, style, shuffle_seed, marker),
                        "labels": [code] if variant else [],
                        "latent_facts": vars(facts),
                        "changed_factor": factor, "style_index": style,
                        "origin": ORIGIN,
                    })
    if len(rows) != COUNTS[split] * 8 or len({r["text"] for r in rows}) != len(rows):
        raise AssertionError("V28 rows are incomplete or duplicated")
    if any(re.search(r"\b(?:D1|R1|A1|P1|G1|S1)\b", row["text"])
           or any(mark in row["text"] for mark in ('"', "“", "”", "\n", "\r"))
           or not 80 <= len(row["text"]) <= 900 for row in rows):
        raise AssertionError("V28 abstract-only card format differs")
    return rows


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()


def prepare(root: Path = ROOT) -> dict:
    root = root.resolve()
    outputs = [root / DATA / f"{split}.jsonl" for split in PUBLIC_SEEDS]
    outputs += [root / COMMITMENT, root / PRIVATE_SEED]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preserve the existing V28 seal")
    rendered = {split: _jsonl(build_rows(split, seed))
                for split, seed in PUBLIC_SEEDS.items()}
    if set(json.loads(line)["text"] for line in rendered["train"].decode().splitlines()) & set(
            json.loads(line)["text"] for line in rendered["dev"].decode().splitlines()):
        raise AssertionError("V28 train and development text overlap")
    seed = secrets.token_hex(24)
    for split, content in rendered.items():
        _write_new(root / DATA / f"{split}.jsonl", content)
    _write_new(root / PRIVATE_SEED, (seed + "\n").encode("ascii"))
    record = {
        "schema_version": 1, "study": "v28_bilingual_evidence_hybrid_synthetic",
        "status": "sealed_same_origin_test_not_yet_generated",
        "scope": "invented_abstract_metadata_only",
        "test_seed_sha256": hashlib.sha256(seed.encode()).hexdigest(),
        "train_sha256": hashlib.sha256(rendered["train"]).hexdigest(),
        "dev_sha256": hashlib.sha256(rendered["dev"]).hexdigest(),
        "generator_sha256": sha256(root / "training/generate_v28_hybrid.py"),
        "source_dependency_sha256": {
            name: sha256(root / name) for name in (
                "training/generate_v25_synthetic.py",
                "training/v25_ro.py", "training/v25_uk.py",
                "training/v26_ro.py", "training/v26_uk.py",
                "training/v28_ro.py", "training/v28_uk.py")},
        "taxonomy_sha256": {lang: taxonomy_sha256(root, lang)
                            for lang in ("ro", "uk")},
        "rows": {split: len(build_rows(split, public_seed))
                 for split, public_seed in PUBLIC_SEEDS.items()},
        "test_rows_at_registration": 0,
        "independent_human_authorship": False,
        "independent_language_review": False,
        "eligible_for_live_child_message_screening": False,
    }
    _write_new(root / COMMITMENT,
               (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    return {"commitment": str(COMMITMENT), "train_rows": 1536,
            "dev_rows": 384, "test_rows_generated": False}


def reveal(root: Path, selection_path: Path) -> dict:
    root = root.resolve()
    relative = selection_path if not selection_path.is_absolute() else selection_path.relative_to(root)
    if relative.is_absolute() or ".." in relative.parts or relative != ARTIFACT / "research.json":
        raise ValueError("V28 selection must be its registered artifact manifest")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Commit the V28 selection before test reveal")
    if subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(relative)],
                      cwd=root, capture_output=True, check=False).returncode:
        raise ValueError("V28 selection is not committed")
    record = json.loads((root / COMMITMENT).read_text())
    selection = json.loads((root / relative).read_text())
    if (selection.get("study") != record.get("study") or
            selection.get("status") != "selected_before_test_reveal" or
            selection.get("test_seed_sha256") != record.get("test_seed_sha256") or
            selection.get("test_seed_commitment_sha256") != sha256(root / COMMITMENT) or
            selection.get("development_eligible_for_test_reveal") is not True):
        raise ValueError("V28 selected model does not satisfy its pretest seal")
    seed = (root / PRIVATE_SEED).read_text(encoding="ascii").strip()
    if hashlib.sha256(seed.encode()).hexdigest() != record["test_seed_sha256"]:
        raise ValueError("V28 private seed differs from its commitment")
    output = root / DATA / "test.jsonl"
    _write_new(output, _jsonl(build_rows("test", seed)))
    return {"test_rows": 384, "test_sha256": sha256(output),
            "same_origin_synthetic_only": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "reveal"))
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    result = (prepare() if args.command == "prepare" else
              reveal(ROOT, args.selection) if args.selection else
              (_ for _ in ()).throw(ValueError("Reveal needs --selection")))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
