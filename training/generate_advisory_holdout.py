"""Fresh Romanian abstract D1/S1 holdout for frozen advisory-model transfer."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "batch-0025"
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILDERS = {code: REFERENCE["BUILDERS"][code] for code in ("D1", "S1")}
FIELDS = REFERENCE["FIELD_NAMES"]
LABELS = {
    "role": "proveniența rolului descris",
    "anchor": "indiciul separat de siguranță sau ajutor",
    "pattern": "apropierea indirectă ori repetată de sprijin",
    "explicit": "sprijinul solicitat direct",
    "field": "componenta de sprijin examinată",
    "field_passed": "componenta examinată respectată",
    "support_ok": "celelalte componente respectate",
}
FRAMES = ("Înregistrare ipotetică a semnalelor", "Inventar fictiv de componente")


def render(facts: dict, seed: int, symbol: str) -> str:
    rng = random.Random(seed)
    pieces = []
    for key, value in facts.items():
        shown = (FIELDS[value] if key == "field" else
                 "are loc" if value is True else "nu are loc" if value is False else str(value))
        pieces.append(f"{LABELS[key]} — {shown}")
    rng.shuffle(pieces)
    return (f"{rng.choice(FRAMES)} {symbol}. " + "; ".join(pieces) +
            ". Sunt doar atribute inventate, nu o conversație.")


def build_rows(batch_id: str = BATCH_ID) -> list[dict]:
    if batch_id != BATCH_ID:
        raise ValueError("Unknown advisory holdout")
    rng = random.Random(825025)
    rows = []
    for code, builder in BUILDERS.items():
        for index in range(24):
            symbol = f"atribut-{batch_id[-4:]}-{len(rows) // 2 + 1:03d}"
            kind, negative, positive, judge = builder(index + 576, rng, symbol)
            if (set(negative) != set(positive) or
                    sum(negative[key] != positive[key] for key in negative) != 1 or
                    judge(negative) or not judge(positive)):
                raise AssertionError(f"Invalid advisory pair: {symbol}")
            seed = rng.randrange(1 << 30)
            for variant, facts in enumerate((negative, positive)):
                rows.append({"id": f"{batch_id}-ro-{len(rows) + 1:04d}", "language": "ro",
                             "source_kind": kind, "split": "test", "text": render(facts, seed, symbol),
                             "labels": [code] if variant else [], "origin": "abstract_advisory_v1"})
    assert len(rows) == 96
    return rows


def main() -> None:
    stem = ROOT / "data/synthetic" / BATCH_ID
    paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Advisory holdout already exists")
    rows = build_rows()
    paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")
    table = ["# batch-0025 — Romanian abstract advisory transfer", "",
             "Stare: în așteptarea validării automate. Split: `test`. 96 de fișe simbolice fără comunicări reale ori realiste.", "",
             "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
    table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                 for row in rows)
    table.extend(("", "Fiecare pereche schimbă un singur atribut verificat de regula tipizată.", ""))
    paths[1].write_text("\n".join(table), encoding="utf-8")
    paths[2].write_text(json.dumps(candidate_manifest(ROOT, BATCH_ID), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"{BATCH_ID}: {len(rows)} test cards")


if __name__ == "__main__":
    main()
