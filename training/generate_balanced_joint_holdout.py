"""Sealed Romanian abstract D1 surfaces for the balanced joint-field study."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILD_D1 = REFERENCE["BUILDERS"]["D1"]
BATCH = "batch-0037"
PAIR_COUNT = 48


def render(facts: dict, seed: int, symbol: str, style: int) -> str:
    if style == 0:
        clauses = [
            "proveniența din registru aparține unui minor" if facts["role"] == "minor" else
            "proveniența din registru aparține unui adult",
            "protecția figurează printre reperele active" if facts["anchor"] else
            "protecția figurează printre reperele inactive",
            "apropierea indirectă sau reluată de ajutor are marcaj pozitiv" if facts["pattern"] else
            "apropierea indirectă sau reluată de ajutor are marcaj negativ",
            "solicitarea explicită de sprijin are marcaj pozitiv" if facts["explicit"] else
            "solicitarea explicită de sprijin are marcaj negativ",
        ]
        intro, tail = "Inventar fictiv cu patru repere", "Nu conține relatare ori replică."
    elif style == 1:
        clauses = [
            "entitatea inițiatoare este încadrată ca minor" if facts["role"] == "minor" else
            "entitatea inițiatoare este încadrată ca adult",
            "atributul contextual de siguranță rămâne valabil" if facts["anchor"] else
            "atributul contextual de siguranță este anulat",
            "revenirea ocolită la tema sprijinului rămâne valabilă" if facts["pattern"] else
            "revenirea ocolită la tema sprijinului este anulată",
            "cererea directă de sprijin rămâne valabilă" if facts["explicit"] else
            "cererea directă de sprijin este anulată",
        ]
        intro, tail = "Fișă de stări inventate pentru control", "Sunt doar valori, nu cuvintele unei persoane."
    else:
        raise ValueError("Unknown balanced joint holdout surface")
    random.Random(seed).shuffle(clauses)
    return f"{intro} {symbol}. " + "; ".join(clauses) + f". {tail}"


def build_d1_facts(batch_id: str) -> list[dict]:
    if batch_id != BATCH:
        raise ValueError("Unknown balanced joint holdout batch")
    facts = []
    for index in range(PAIR_COUNT):
        rng = random.Random(2301 + index * 31)
        _, negative, positive, judge = BUILD_D1(2301 + index, rng, "unused-symbol")
        if (set(negative) != set(positive) or
                sum(negative[key] != positive[key] for key in negative) != 1 or
                judge(negative) or not judge(positive)):
            raise AssertionError("Invalid balanced joint one-fact pair")
        facts.extend((negative, positive))
    return facts


def build_rows(batch_id: str) -> list[dict]:
    facts = build_d1_facts(batch_id)
    seeds = random.Random(537037)
    rows = []
    for index, (negative, positive) in enumerate(zip(facts[::2], facts[1::2]), 1):
        symbol = f"valori-{index:03d}-{batch_id[-4:]}"
        seed = seeds.randrange(1 << 30)
        style = 0 if index <= PAIR_COUNT // 2 else 1
        for variant, item in enumerate((negative, positive)):
            rows.append({
                "id": f"{batch_id}-ro-{len(rows) + 1:05d}", "language": "ro",
                "source_kind": "message", "split": "test",
                "text": render(item, seed, symbol, style),
                "labels": ["D1"] if variant else [],
                "origin": "abstract_balanced_joint_holdout_v1",
            })
    return rows


def main() -> None:
    stem = ROOT / "data/synthetic" / BATCH
    paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Preserve existing balanced joint holdout")
    rows = build_rows(BATCH)
    paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")
    table = ["# batch-0037 — fișe abstracte pentru controlul clasificării echilibrate", "",
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
