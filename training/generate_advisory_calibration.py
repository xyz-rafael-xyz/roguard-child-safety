"""Two fresh, abstract Romanian surfaces for prospective V17 cutoff transfer."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILDERS = {code: REFERENCE["BUILDERS"][code] for code in ("D1", "S1")}
FIELD_NAMES = REFERENCE["FIELD_NAMES"]
SPECS = {"batch-0031": ("dev", 1201), "batch-0032": ("test", 1409)}


def render(code: str, facts: dict, seed: int, symbol: str, split: str) -> str:
    if code == "D1":
        if split == "dev":
            clauses = [
                "sursa din inventar aparține unui minor" if facts["role"] == "minor" else
                "sursa din inventar aparține unui adult",
                "există o referință la siguranță sau sprijin" if facts["anchor"] else
                "nu există referința la siguranță sau sprijin",
                "este marcată apropierea indirectă sau repetată" if facts["pattern"] else
                "nu este marcată apropierea indirectă sau repetată",
                "apare o cerere directă de sprijin" if facts["explicit"] else
                "nu apare o cerere directă de sprijin",
            ]
        else:
            clauses = [
                "rolul emitentului corespunde unui minor" if facts["role"] == "minor" else
                "rolul emitentului corespunde unui adult",
                "variabila de context păstrează legătura cu protecția" if facts["anchor"] else
                "variabila de context exclude legătura cu protecția",
                "variabila de traseu atestă revenirea indirectă" if facts["pattern"] else
                "variabila de traseu exclude revenirea indirectă",
                "variabila de solicitare confirmă sprijinul cerut direct" if facts["explicit"] else
                "variabila de solicitare infirmă sprijinul cerut direct",
            ]
    elif split == "dev":
        clauses = [
            f"componenta evaluată este {FIELD_NAMES[facts['field']]}",
            "această componentă îndeplinește cerința" if facts["field_passed"] else
            "această componentă nu îndeplinește cerința",
            "celelalte verificări sunt satisfăcute" if facts["support_ok"] else
            "celelalte verificări nu sunt satisfăcute",
        ]
    else:
        clauses = [
            f"rubrica analizată privește {FIELD_NAMES[facts['field']]}",
            "rezultatul rubricii este conform" if facts["field_passed"] else
            "rezultatul rubricii este neconform",
            "restul condițiilor sunt respectate" if facts["support_ok"] else
            "restul condițiilor sunt încălcate",
        ]
    random.Random(seed).shuffle(clauses)
    if split == "dev":
        return f"Inventar de proprietăți imaginare {symbol}. " + "; ".join(clauses) + ". Fără dialog sau răspuns redat."
    return f"Schema de control ipotetic {symbol}: " + "; ".join(clauses) + ". Conține exclusiv stări inventate."


def build_rows(batch_id: str) -> list[dict]:
    if batch_id not in SPECS:
        raise ValueError("Unknown advisory calibration batch")
    split, offset = SPECS[batch_id]
    rng = random.Random(931000 + int(batch_id[-4:]))
    rows = []
    for code, builder in BUILDERS.items():
        for index in range(24):
            symbol = f"schemă-{batch_id[-4:]}-{len(rows) // 2 + 1:04d}"
            state_rng = random.Random(offset + index * 19 + (0 if code == "D1" else 311))
            kind, negative, positive, judge = builder(index + offset, state_rng, symbol)
            if (set(negative) != set(positive) or
                    sum(negative[key] != positive[key] for key in negative) != 1 or
                    judge(negative) or not judge(positive)):
                raise AssertionError(f"Invalid calibration pair: {symbol}")
            seed = rng.randrange(1 << 30)
            for variant, facts in enumerate((negative, positive)):
                rows.append({
                    "id": f"{batch_id}-ro-{len(rows) + 1:05d}", "language": "ro",
                    "source_kind": kind, "split": split,
                    "text": render(code, facts, seed, symbol, split),
                    "labels": [code] if variant else [], "origin": "abstract_advisory_calibration_v1",
                })
    assert len(rows) == 96
    return rows


def main() -> None:
    for batch_id, (split, _) in SPECS.items():
        stem = ROOT / "data/synthetic" / batch_id
        paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
        if any(path.exists() for path in paths):
            raise FileExistsError("Preserve existing calibration batch")
        rows = build_rows(batch_id)
        paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                    for row in rows), encoding="utf-8")
        table = [f"# {batch_id} — fișe abstracte pentru verificarea pragului", "",
                 f"Stare: în așteptarea validării automate. Split: `{split}`. {len(rows)} fișe simbolice fără comunicări reale ori realiste.", "",
                 "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
        table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                     for row in rows)
        table.extend(("", "Perechile adiacente au o singură diferență verificată prin regula tipizată.", ""))
        paths[1].write_text("\n".join(table), encoding="utf-8")
        paths[2].write_text(json.dumps(candidate_manifest(ROOT, batch_id), ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"{batch_id}: {len(rows)} {split} cards")


if __name__ == "__main__":
    main()
