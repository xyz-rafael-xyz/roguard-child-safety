"""Sealed Romanian abstract D1 transfer surfaces for the NLI-adapter study."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILD_D1 = REFERENCE["BUILDERS"]["D1"]
BATCH = "batch-0038"
PAIR_COUNT = 48


def render(facts: dict, seed: int, symbol: str, style: int) -> str:
    if style == 0:
        clauses = [
            "în dreptul originii este înscrisă o persoană sub 18 ani" if facts["role"] == "minor" else
            "în dreptul originii este înscrisă o persoană majoră",
            "criteriul de siguranță sau sprijin apare ca disponibil" if facts["anchor"] else
            "criteriul de siguranță sau sprijin apare ca indisponibil",
            "apropierea indirectă ori reiterată de sprijin apare ca prezentă" if facts["pattern"] else
            "apropierea indirectă ori reiterată de sprijin apare ca absentă",
            "solicitarea explicită de ajutor apare ca prezentă" if facts["explicit"] else
            "solicitarea explicită de ajutor apare ca absentă",
        ]
        intro, tail = "Inventar ipotetic de criterii", "Nu redă cuvintele vreunei persoane."
    elif style == 1:
        clauses = [
            "clasificarea inițiatorului: minor" if facts["role"] == "minor" else
            "clasificarea inițiatorului: adult",
            "indicatorul contextual de protecție: confirmat" if facts["anchor"] else
            "indicatorul contextual de protecție: infirmat",
            "indicatorul reluării ocolite a căutării de ajutor: confirmat" if facts["pattern"] else
            "indicatorul reluării ocolite a căutării de ajutor: infirmat",
            "indicatorul cererii directe de sprijin: confirmat" if facts["explicit"] else
            "indicatorul cererii directe de sprijin: infirmat",
        ]
        intro, tail = "Fișă artificială de verificare", "Conține doar atribute, nu un dialog."
    else:
        raise ValueError("Unknown NLI transfer surface")
    random.Random(seed).shuffle(clauses)
    return f"{intro} {symbol}. " + "; ".join(clauses) + f". {tail}"


def build_d1_facts(batch_id: str) -> list[dict]:
    if batch_id != BATCH:
        raise ValueError("Unknown NLI transfer batch")
    values = []
    for index in range(PAIR_COUNT):
        rng = random.Random(2701 + index * 37)
        kind, negative, positive, judge = BUILD_D1(2701 + index, rng, "unused-symbol")
        if (kind != "message" or set(negative) != set(positive) or
                sum(negative[key] != positive[key] for key in negative) != 1 or
                judge(negative) or not judge(positive)):
            raise AssertionError("Invalid one-fact NLI transfer pair")
        values.extend((negative, positive))
    return values


def build_rows(batch_id: str) -> list[dict]:
    facts = build_d1_facts(batch_id)
    seeds = random.Random(538038)
    rows = []
    for index, (negative, positive) in enumerate(zip(facts[::2], facts[1::2]), 1):
        symbol = f"criterii-{index:03d}-{batch_id[-4:]}"
        seed = seeds.randrange(1 << 30)
        style = 0 if index <= PAIR_COUNT // 2 else 1
        for variant, item in enumerate((negative, positive)):
            rows.append({
                "id": f"{batch_id}-ro-{len(rows) + 1:05d}", "language": "ro",
                "source_kind": "message", "split": "test",
                "text": render(item, seed, symbol, style),
                "labels": ["D1"] if variant else [],
                "origin": "abstract_nli_transfer_holdout_v1",
            })
    return rows


def main() -> None:
    stem = ROOT / "data/synthetic" / BATCH
    paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Preserve existing NLI transfer holdout")
    rows = build_rows(BATCH)
    paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")
    table = ["# batch-0038 — fișe abstracte pentru transferul modelului NLI", "",
             "Stare: în așteptarea validării automate. Split: `test`. 96 fișe simbolice fără comunicări reale ori realiste.", "",
             "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
    table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                 for row in rows)
    table.extend(("", "Perechile adiacente au o singură diferență verificată prin regula tipizată.", ""))
    paths[1].write_text("\n".join(table), encoding="utf-8")
    paths[2].write_text(json.dumps(candidate_manifest(ROOT, BATCH), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"{BATCH}: {len(rows)} sealed abstract test cards")


if __name__ == "__main__":
    main()
