"""Prospective same-origin paired metadata with latent facts for factorized learning."""

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
from training import v26_ro, v26_uk
from training.generate_v25_synthetic import (_d1_oracle, _d1_pair, _rng,
                                             _s1_oracle, _s1_pair, _write_new)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/synthetic/v26")
COMMITMENT = Path("eval/prospective/v26-facts-seed-commitment.json")
PRIVATE_SEED = Path("review_runs/v26-facts-test-seed.txt")
ARTIFACT = Path("models/bi-mmbert-v26-facts")
LANGUAGE = {"ro": v26_ro, "uk": v26_uk}
COUNTS = {"train": 192, "dev": 48, "test": 48}
PUBLIC_SEEDS = {"train": "v26-facts-train-2026-10-01",
                "dev": "v26-facts-dev-2026-10-01"}
ORIGIN = "v26_same_origin_exploratory_latent_facts_v1"


def _style(split: str, index: int, factor_count: int) -> int:
    if split == "train":
        return (index // factor_count) % 7
    if split == "dev":
        return 7
    return 8 + (index // factor_count) % 2


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
        raise ValueError("V26 needs a registered split and nontrivial seed")
    rows = []
    for language in ("ro", "uk"):
        for code in ("D1", "S1"):
            for index in range(COUNTS[split]):
                rng = _rng(seed, language, code, index)
                factor, negative, positive = (_d1_pair(index, rng) if code == "D1"
                                              else _s1_pair(index))
                style = _style(split, index, 4 if code == "D1" else 2)
                marker = hashlib.sha256(
                    f"v26-marker:{seed}:{language}:{code}:{index}".encode()
                ).hexdigest()[:10]
                shuffle_seed = rng.randrange(1 << 60)
                for variant, facts in enumerate((negative, positive)):
                    truth = _d1_oracle(facts) if code == "D1" else _s1_oracle(facts)
                    if truth != bool(variant):
                        raise AssertionError("V26 latent facts and policy oracle differ")
                    rows.append({
                        "id": f"v26-{split}-{language}-{code.lower()}-{index:04d}-{variant}",
                        "language": language,
                        "source_kind": "message" if code == "D1" else "response",
                        "split": split,
                        "text": _render(language, code, facts, style, shuffle_seed, marker),
                        "labels": [code] if variant else [],
                        "latent_facts": vars(facts),
                        "changed_factor": factor,
                        "style_index": style,
                        "origin": ORIGIN,
                    })
    if len(rows) != COUNTS[split] * 8 or len({r["text"] for r in rows}) != len(rows):
        raise AssertionError("V26 rows are incomplete or duplicated")
    if any(re.search(r"\b(?:D1|R1|A1|P1|G1|S1)\b", row["text"])
           or any(mark in row["text"] for mark in ('"', "“", "”", "\n", "\r"))
           or not 80 <= len(row["text"]) <= 900 for row in rows):
        raise AssertionError("V26 abstract metadata format differs")
    return rows


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()


def prepare(root: Path = ROOT) -> dict:
    root = root.resolve()
    outputs = [root / DATA / f"{split}.jsonl" for split in PUBLIC_SEEDS]
    outputs += [root / COMMITMENT, root / PRIVATE_SEED]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preserve the existing V26 seal")
    rendered = {split: _jsonl(build_rows(split, seed))
                for split, seed in PUBLIC_SEEDS.items()}
    if set(json.loads(line)["text"] for line in rendered["train"].decode().splitlines()) & set(
            json.loads(line)["text"] for line in rendered["dev"].decode().splitlines()):
        raise AssertionError("V26 train and development text overlap")
    seed = secrets.token_hex(24)
    for split, content in rendered.items():
        _write_new(root / DATA / f"{split}.jsonl", content)
    _write_new(root / PRIVATE_SEED, (seed + "\n").encode("ascii"))
    record = {
        "schema_version": 1, "study": "v26_bilingual_factorized_synthetic",
        "status": "sealed_same_origin_test_not_yet_generated",
        "scope": "invented_abstract_metadata_only",
        "test_seed_sha256": hashlib.sha256(seed.encode("ascii")).hexdigest(),
        "train_sha256": hashlib.sha256(rendered["train"]).hexdigest(),
        "dev_sha256": hashlib.sha256(rendered["dev"]).hexdigest(),
        "generator_sha256": sha256(root / "training/generate_v26_facts.py"),
        "shared_oracle_sha256": sha256(root / "training/generate_v25_synthetic.py"),
        "wording_sha256": {language: sha256(root / f"training/v26_{language}.py")
                           for language in LANGUAGE},
        "prior_wording_sha256": {language: sha256(root / f"training/v25_{language}.py")
                                 for language in LANGUAGE},
        "taxonomy_sha256": {language: taxonomy_sha256(root, language)
                            for language in LANGUAGE},
        "rows": {split: len(build_rows(split, public_seed))
                 for split, public_seed in PUBLIC_SEEDS.items()},
        "test_rows_at_registration": 0,
        "independent_human_authorship": False,
        "independent_language_review": False,
        "eligible_for_live_child_message_screening": False,
    }
    _write_new(root / COMMITMENT,
               (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    return {"commitment": str(COMMITMENT), "train_rows": len(build_rows("train", PUBLIC_SEEDS["train"])),
            "dev_rows": len(build_rows("dev", PUBLIC_SEEDS["dev"])), "test_rows_generated": False}


def reveal(root: Path, selection_path: Path) -> dict:
    root = root.resolve()
    relative = selection_path if not selection_path.is_absolute() else selection_path.relative_to(root)
    if relative.is_absolute() or ".." in relative.parts or relative != ARTIFACT / "research.json":
        raise ValueError("V26 selection must be its registered artifact manifest")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Commit the V26 selection before test reveal")
    if subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(relative)],
                      cwd=root, capture_output=True, check=False).returncode:
        raise ValueError("V26 selection is not committed")
    record = json.loads((root / COMMITMENT).read_text())
    selection = json.loads((root / relative).read_text())
    if (selection.get("study") != record.get("study") or
            selection.get("status") != "selected_before_test_reveal" or
            selection.get("test_seed_sha256") != record.get("test_seed_sha256") or
            selection.get("test_seed_commitment_sha256") != sha256(root / COMMITMENT) or
            selection.get("development_eligible_for_test_reveal") is not True):
        raise ValueError("V26 selected model does not bind its seal")
    seed = (root / PRIVATE_SEED).read_text(encoding="ascii").strip()
    if hashlib.sha256(seed.encode()).hexdigest() != record["test_seed_sha256"]:
        raise ValueError("V26 private seed differs from its commitment")
    output = root / DATA / "test.jsonl"
    _write_new(output, _jsonl(build_rows("test", seed)))
    return {"test_rows": COUNTS["test"] * 8, "test_sha256": sha256(output),
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
