"""Generate controlled, fictional bilingual support-signal pairs.

All stems are locally authored for this study. Do not ingest external text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "synthetic" / "v3_message_banks.json"
SPLIT_COUNTS = {"train": 20, "dev": 6, "test": 6}
TRAIN_PREFIXES = {
    "ro": ("", "Bună, ", "Hei, ", "Salut, ", "Scuze, ", "Am o întrebare. ",
           "Poți să citești asta? ", "Nu știu cum să încep. ",
           "E ceva ce vreau să spun. ", "Azi voiam să întreb ceva. ",
           "Ok, ", "Uite, ", "Mă poți asculta? ",
           "Nu știu dacă întreb bine. ", "Te rog să mă asculți: ",
           "Aș vrea un răspuns. "),
    "uk": ("", "Привіт, ", "Вітаю, ", "Гей, ", "Вибач, ", "Маю запитання. ",
           "Можеш це прочитати? ", "Не знаю, як почати. ",
           "Хочу дещо сказати. ", "Сьогодні хотів запитати. ",
           "Ну, ", "Дивись, ", "Можеш мене вислухати? ",
           "Не знаю, чи правильно питаю. ", "Будь ласка, вислухай: ",
           "Хотів би отримати відповідь. ")
}
DEV_PREFIXES = {
    "ro": ("", "Mă ajuți cu o întrebare? ", "Nu prea știu cum să explic. ", "Bună dimineața, "),
    "uk": ("", "Допоможеш із запитанням? ", "Не дуже знаю, як пояснити. ", "Доброго ранку, ")
}
TEST_PREFIXES = {
    "ro": ("", "Salut, am nevoie să întreb ceva. ", "Poate sună ciudat, dar ", "Aș vrea să vorbim. "),
    "uk": ("", "Привіт, хочу дещо запитати. ", "Можливо, це дивно звучить, але ", "Хотів би поговорити. ")
}


def _render(prefix: str, stem: str, style: int) -> str:
    base = stem.rstrip(".!?…")
    variant = (stem, stem.lower(), base, base + "…")[style]
    return unicodedata.normalize("NFC", prefix + variant)


def _fingerprint(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split()).rstrip(".!?…")


def generate(split: str) -> list[dict]:
    if split not in SPLIT_COUNTS:
        raise ValueError("Unknown split")
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    if set(bank) != {"ro", "uk"}:
        raise ValueError("Expected Romanian and Ukrainian banks")
    rows = []
    for language in ("ro", "uk"):
        if set(bank[language]) != set(SPLIT_COUNTS):
            raise ValueError("Split bank mismatch")
        if len(bank[language][split]) != SPLIT_COUNTS[split]:
            raise ValueError("Unexpected stem count")
        prefixes = {"train": TRAIN_PREFIXES, "dev": DEV_PREFIXES,
                    "test": TEST_PREFIXES}[split][language]
        seen_families = set()
        for family, positive, negative in bank[language][split]:
            if (family in seen_families or not all(isinstance(value, str) and value
                    for value in (family, positive, negative)) or positive == negative):
                raise ValueError("Invalid or duplicate family")
            seen_families.add(family)
            for index, prefix in enumerate(prefixes):
                for style in range(4):
                    pair_id = f"{language}-{split}-{family}-{index:02d}-{style}"
                    for label, stem in ((1, positive), (0, negative)):
                        text = _render(prefix, stem, style)
                        if not 15 <= len(text) <= 250 or re.search(
                                r"https?://|\b\S+@\S+\b|\b\d{7,}\b", text):
                            raise ValueError("Text shape or private identifier failed")
                        rows.append({"id": hashlib.sha256(
                            f"{pair_id}|{label}".encode()).hexdigest()[:20],
                            "pair_id": pair_id, "language": language,
                            "split": split, "family": family, "label": label,
                            "text": text})
    random.Random({"train": 3101, "dev": 3102, "test": 3103}[split]).shuffle(rows)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate row ID")
    return rows


def validate_splits() -> dict:
    all_rows = {split: generate(split) for split in SPLIT_COUNTS}
    groups = {split: {_fingerprint(row["text"]) for row in rows}
              for split, rows in all_rows.items()}
    for first in SPLIT_COUNTS:
        for second in SPLIT_COUNTS:
            if first < second and groups[first] & groups[second]:
                raise ValueError("Normalized text overlap across splits")
    for language in ("ro", "uk"):
        families = {split: {row["family"] for row in all_rows[split]
                            if row["language"] == language}
                    for split in SPLIT_COUNTS}
        if any(families[first] & families[second]
               for first in SPLIT_COUNTS for second in SPLIT_COUNTS if first < second):
            raise ValueError("Family overlap across splits")
    for split, rows in all_rows.items():
        if len({row["text"] for row in rows}) != len(rows):
            raise ValueError("Duplicate exact text within split")
        for language in ("ro", "uk"):
            subset = [row for row in rows if row["language"] == language]
            expected = SPLIT_COUNTS[split] * len({"train": TRAIN_PREFIXES,
                  "dev": DEV_PREFIXES, "test": TEST_PREFIXES}[split][language]) * 4
            if (sum(row["label"] for row in subset) != expected or
                    len(subset) != 2 * expected):
                raise ValueError("Unbalanced pair count")
    return {split: {"rows": len(rows), "pairs": len(rows) // 2,
                    "sha256": hashlib.sha256("".join(json.dumps(row, ensure_ascii=False,
                        sort_keys=True) + "\n" for row in rows).encode()).hexdigest()}
            for split, rows in all_rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fictional bilingual message pairs")
    parser.add_argument("--split", choices=tuple(SPLIT_COUNTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_splits()
    if args.output.exists():
        parser.error("Output already exists")
    rows = generate(args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False,
        sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"split": args.split, **manifest[args.split]}, sort_keys=True))


if __name__ == "__main__":
    main()
