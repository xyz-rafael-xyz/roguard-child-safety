"""Build a sealed, same-origin bilingual abstract-card pilot.

This exploratory generator is deliberately outside ``load_approved``. Its
Ukrainian text is still a draft, and its cards are not child messages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import secrets
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from roguard.policy import (DisclosureEvidenceCard, SupportContractCard,
                            check_disclosure_evidence, check_support_contract)
from roguard.review import sha256, taxonomy_sha256
from training import v25_ro, v25_uk

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/synthetic/v25")
COMMITMENT = Path("eval/prospective/v25-synthetic-seed-commitment.json")
PRIVATE_SEED = Path("review_runs/v25-synthetic-test-seed.txt")
LANGUAGE = {"ro": v25_ro, "uk": v25_uk}
COUNTS = {"train": 192, "dev": 48, "test": 48}  # pairs per language and category
PUBLIC_SEEDS = {"train": "v25-train-2026-10-01", "dev": "v25-dev-2026-10-01"}
STYLE_INDICES = {"train": (0, 1, 2, 3), "dev": (4,), "test": (5, 6)}
D1_FACTORS = ("source_role", "safety_or_support_anchor",
              "indirect_or_repeated_support_pattern", "explicit_support_request")
S1_FACTORS = ("applicable", "passed")
FIELDS = tuple(sorted(v25_ro.FIELDS))
ORIGIN = "v25_same_origin_exploratory_abstract_v1"


@dataclass(frozen=True)
class D1Facts:
    source_role: str
    safety_or_support_anchor: bool
    indirect_or_repeated_support_pattern: bool
    explicit_support_request: bool


@dataclass(frozen=True)
class S1Facts:
    field: str
    applicable: bool
    passed: bool


def _rng(seed: str, language: str, code: str, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{language}:{code}:{index}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _d1_pair(index: int, rng: random.Random) -> tuple[str, D1Facts, D1Facts]:
    factor = D1_FACTORS[index % len(D1_FACTORS)]
    indirect = bool((index // len(D1_FACTORS)) % 2)
    positive = D1Facts("minor", True, indirect, not indirect)
    if factor == "source_role":
        negative = replace(positive, source_role="adult")
    elif factor == "safety_or_support_anchor":
        negative = replace(positive, safety_or_support_anchor=False)
    elif factor == "indirect_or_repeated_support_pattern":
        positive = replace(positive, indirect_or_repeated_support_pattern=True,
                           explicit_support_request=False)
        negative = replace(positive, indirect_or_repeated_support_pattern=False)
    else:
        positive = replace(positive, indirect_or_repeated_support_pattern=False,
                           explicit_support_request=True)
        negative = replace(positive, explicit_support_request=False)
    assert not _d1_oracle(negative) and _d1_oracle(positive)
    assert sum(getattr(negative, key) != getattr(positive, key)
               for key in D1_FACTORS) == 1
    return factor, negative, positive


def _d1_oracle(facts: D1Facts) -> bool:
    return bool(check_disclosure_evidence(DisclosureEvidenceCard(**vars(facts))).reason_codes)


def _s1_oracle(facts: S1Facts) -> bool:
    other = "acknowledgement" if facts.field != "acknowledgement" else "next_step"
    applicable = {other} | ({facts.field} if facts.applicable else set())
    passed = {other} | ({facts.field} if facts.applicable and facts.passed else set())
    return bool(check_support_contract(SupportContractCard(
        frozenset(applicable), frozenset(passed))).reason_codes)


def _s1_pair(index: int) -> tuple[str, S1Facts, S1Facts]:
    field = FIELDS[(index // len(S1_FACTORS)) % len(FIELDS)]
    factor = S1_FACTORS[index % len(S1_FACTORS)]
    positive = S1Facts(field, True, False)
    negative = (replace(positive, applicable=False) if factor == "applicable"
                else replace(positive, passed=True))
    assert not _s1_oracle(negative) and _s1_oracle(positive)
    assert sum(getattr(negative, key) != getattr(positive, key)
               for key in S1_FACTORS) == 1
    return factor, negative, positive


def _style(split: str, index: int, factor_count: int) -> int:
    choices = STYLE_INDICES[split]
    return choices[(index // factor_count) % len(choices)]


def _render(language: str, code: str, facts: D1Facts | S1Facts,
            style_index: int, rng: random.Random, marker: str) -> str:
    module = LANGUAGE[language]
    style = (module.D1_STYLES if code == "D1" else module.S1_STYLES)[style_index]
    if code == "D1":
        assert isinstance(facts, D1Facts)
        clauses = [
            style["role"][facts.source_role != "minor"],
            style["anchor"][not facts.safety_or_support_anchor],
            style["indirect"][not facts.indirect_or_repeated_support_pattern],
            style["explicit"][not facts.explicit_support_request],
        ]
    else:
        assert isinstance(facts, S1Facts)
        clauses = [
            style["field"].format(module.FIELDS[facts.field]),
            style["applicable"][not facts.applicable],
            style["passed"][not facts.passed],
        ]
    rng.shuffle(clauses)
    return f"{style['open']} {marker}: " + "; ".join(clauses) + f". {style['close']}"


def build_rows(split: str, seed: str) -> list[dict]:
    if split not in COUNTS or not isinstance(seed, str) or len(seed) < 16:
        raise ValueError("Choose a registered split and a nontrivial seed")
    rows = []
    for language in ("ro", "uk"):
        for code in ("D1", "S1"):
            for index in range(COUNTS[split]):
                rng = _rng(seed, language, code, index)
                factor, negative, positive = (_d1_pair(index, rng) if code == "D1"
                                              else _s1_pair(index))
                style_index = _style(split, index, len(D1_FACTORS if code == "D1"
                                                       else S1_FACTORS))
                marker = hashlib.sha256(
                    f"marker:{seed}:{language}:{code}:{index}".encode("utf-8")
                ).hexdigest()[:10]
                shuffle_seed = rng.randrange(1 << 60)
                for variant, facts in enumerate((negative, positive)):
                    row = {
                        "id": f"v25-{split}-{language}-{code.lower()}-{index:04d}-{variant}",
                        "language": language,
                        "source_kind": "message" if code == "D1" else "response",
                        "split": split,
                        "text": _render(language, code, facts, style_index,
                                        random.Random(shuffle_seed), marker),
                        "labels": [code] if variant else [],
                        "origin": ORIGIN,
                    }
                    rows.append(row)
    if len(rows) != COUNTS[split] * 8 or len({row["text"] for row in rows}) != len(rows):
        raise AssertionError("V25 pair count or text uniqueness differs")
    if any(re.search(r"\b(?:D1|R1|A1|P1|G1|S1)\b", row["text"])
           or any(mark in row["text"] for mark in ('"', "“", "”", "\n", "\r"))
           or not 80 <= len(row["text"]) <= 900 for row in rows):
        raise AssertionError("V25 abstract-only card format differs")
    return rows


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode("utf-8")


def prepare(root: Path = ROOT) -> dict:
    root = root.resolve()
    outputs = [root / DATA / f"{split}.jsonl" for split in PUBLIC_SEEDS]
    outputs += [root / COMMITMENT, root / PRIVATE_SEED]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preserve the existing V25 preparation or private test seed")
    rendered = {split: _jsonl(build_rows(split, seed))
                for split, seed in PUBLIC_SEEDS.items()}
    if set(json.loads(line)["text"] for line in rendered["train"].decode().splitlines()) & set(
            json.loads(line)["text"] for line in rendered["dev"].decode().splitlines()):
        raise AssertionError("V25 train and development overlap")
    seed = secrets.token_hex(24)
    _write_new(root / PRIVATE_SEED, (seed + "\n").encode("ascii"))
    for split, content in rendered.items():
        _write_new(root / DATA / f"{split}.jsonl", content)
    record = {
        "schema_version": 1,
        "status": "sealed_same_origin_synthetic_test_not_yet_generated",
        "scope": "invented_abstract_metadata_only",
        "test_seed_sha256": hashlib.sha256(seed.encode("ascii")).hexdigest(),
        "train_sha256": hashlib.sha256(rendered["train"]).hexdigest(),
        "dev_sha256": hashlib.sha256(rendered["dev"]).hexdigest(),
        "generator_sha256": sha256(root / "training/generate_v25_synthetic.py"),
        "romanian_wording_sha256": sha256(root / "training/v25_ro.py"),
        "ukrainian_wording_sha256": sha256(root / "training/v25_uk.py"),
        "taxonomy_sha256": {language: taxonomy_sha256(root, language)
                            for language in LANGUAGE},
        "rows": {split: len(build_rows(split, seed)) for split, seed in PUBLIC_SEEDS.items()},
        "test_rows_at_registration": 0,
        "independent_human_authorship": False,
        "independent_language_review": False,
        "eligible_for_live_child_message_screening": False,
    }
    _write_new(root / COMMITMENT,
               (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    return {"commitment": str(COMMITMENT), "test_seed_sha256": record["test_seed_sha256"],
            "train_rows": record["rows"]["train"], "dev_rows": record["rows"]["dev"],
            "test_rows_generated": False}


def reveal(root: Path, selection_path: Path) -> dict:
    """Generate test labels only after the exact selected model is committed."""
    root = root.resolve()
    relative = selection_path if not selection_path.is_absolute() else selection_path.relative_to(root)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
        raise ValueError("Selection record must be a repository JSON file")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Commit the selected model before revealing the test seed")
    if subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(relative)],
                      cwd=root, capture_output=True, check=False).returncode != 0:
        raise ValueError("Selection record is not committed")
    record = json.loads((root / COMMITMENT).read_text(encoding="utf-8"))
    selection = json.loads((root / relative).read_text(encoding="utf-8"))
    if (selection.get("study") != "v25_bilingual_synthetic" or
            selection.get("test_seed_sha256") != record["test_seed_sha256"] or
            selection.get("status") != "selected_before_test_reveal"):
        raise ValueError("Selected model does not bind the sealed V25 test")
    seed = (root / PRIVATE_SEED).read_text(encoding="ascii").strip()
    if hashlib.sha256(seed.encode("ascii")).hexdigest() != record["test_seed_sha256"]:
        raise ValueError("Private test seed differs from its committed commitment")
    output = root / DATA / "test.jsonl"
    _write_new(output, _jsonl(build_rows("test", seed)))
    return {"test_sha256": sha256(output), "test_rows": COUNTS["test"] * 8,
            "selection_sha256": sha256(root / relative),
            "same_origin_synthetic_only": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "reveal"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    try:
        result = (prepare(args.root) if args.command == "prepare" else
                  reveal(args.root, args.selection) if args.selection else
                  (_ for _ in ()).throw(ValueError("Reveal requires --selection")))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, TypeError, ValueError, AssertionError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
