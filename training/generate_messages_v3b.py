"""Second, prospective fictional-message split after the v3 colloquial probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from training.generate_messages_v3 import BANK, ROOT, TRAIN_PREFIXES, _render

NEW_BANK = ROOT / "data/synthetic/v3b_natural_banks.json"
PROBE_BANK = ROOT / "data/synthetic/v3_colloquial_probe.json"


def generate(split: str) -> list[dict]:
    if split not in ("train", "dev", "test"):
        raise ValueError("Unknown split")
    old = json.loads(BANK.read_text(encoding="utf-8"))
    probe = json.loads(PROBE_BANK.read_text(encoding="utf-8"))
    new = json.loads(NEW_BANK.read_text(encoding="utf-8"))
    rows = []
    for language in ("ro", "uk"):
        if split == "train":
            pairs = [(f"old-{old_split}-{index}", positive, negative)
                     for old_split in ("train", "dev", "test")
                     for index, (_, positive, negative) in enumerate(old[language][old_split])]
            pairs += [(f"probe-{index}", positive, negative)
                      for index, (positive, negative) in enumerate(probe[language])]
            prefixes = TRAIN_PREFIXES[language]
            styles = range(4)
        else:
            pairs = [(f"new-{index}", positive, negative)
                     for index, (positive, negative) in enumerate(new[language][split])]
            prefixes = ("",)
            styles = range(1)
        if len(pairs) != (42 if split == "train" else 8):
            raise ValueError("Unexpected stem count")
        for family, positive, negative in pairs:
            if positive == negative:
                raise ValueError("Identical pair members")
            for index, prefix in enumerate(prefixes):
                for style in styles:
                    pair_id = f"v3b-{language}-{split}-{family}-{index}-{style}"
                    for label, stem in ((1, positive), (0, negative)):
                        rows.append({"id": hashlib.sha256(f"{pair_id}|{label}".encode())
                                     .hexdigest()[:20], "pair_id": pair_id,
                                     "language": language, "split": split,
                                     "family": family, "label": label,
                                     "text": _render(prefix, stem, style)})
    random.Random({"train": 3201, "dev": 3202, "test": 3203}[split]).shuffle(rows)
    if len({row["text"].lower().rstrip(".!?…") for row in rows}) != len(rows):
        # Case/punctuation variants intentionally collide under normalization on train.
        if split != "train":
            raise ValueError("Duplicate independent natural stems")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    rows = generate(args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                   for row in rows), encoding="utf-8")
    print(json.dumps({"split": args.split, "rows": len(rows),
                      "pairs": len(rows) // 2,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
