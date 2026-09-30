"""Fresh Romanian abstract surfaces for the factorized D1 study."""

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
SPECS = {"batch-0033": ("dev", 1501, 24, 24),
         "batch-0034": ("test", 1709, 48, 24)}


def render(code: str, facts: dict, seed: int, symbol: str, style: int) -> str:
    if code == "D1":
        clauses = (
            (
                ("entitatea-sursă este un minor", "entitatea-sursă este un adult"),
                ("este prezent un context de siguranță ori sprijin", "contextul de siguranță ori sprijin lipsește"),
                ("apropierea de ajutor are un traseu indirect sau repetat", "apropierea de ajutor nu are traseu indirect sau repetat"),
                ("există solicitare explicită de sprijin", "nu există solicitare explicită de sprijin"),
            ),
            (
                ("fișa identifică un emitent minor", "fișa identifică un emitent adult"),
                ("referința la protecție se regăsește între atribute", "referința la protecție nu se regăsește între atribute"),
                ("este consemnată revenirea ocolită către ajutor", "revenirea ocolită către ajutor este exclusă"),
                ("cererea deschisă de ajutor este confirmată", "cererea deschisă de ajutor este infirmată"),
            ),
            (
                ("statutul celui care inițiază este minor", "statutul celui care inițiază este adult"),
                ("componenta contextuală păstrează un indiciu de sprijin", "componenta contextuală elimină indiciul de sprijin"),
                ("semnalul ocolit revine în structură", "semnalul ocolit nu revine în structură"),
                ("solicitarea directă figurează în schemă", "solicitarea directă nu figurează în schemă"),
            ),
        )[style]
        values = (facts["role"] == "minor", facts["anchor"], facts["pattern"], facts["explicit"])
        parts = [choice[not value] for choice, value in zip(clauses, values)]
    else:
        clauses = (
            (f"componenta examinată este {FIELD_NAMES[facts['field']]}",
             "componenta examinată respectă cerința" if facts["field_passed"] else
             "componenta examinată nu respectă cerința",
             "celelalte condiții de sprijin sunt îndeplinite" if facts["support_ok"] else
             "celelalte condiții de sprijin sunt neîndeplinite"),
            (f"proprietatea vizată privește {FIELD_NAMES[facts['field']]}",
             "proprietatea vizată are stare conformă" if facts["field_passed"] else
             "proprietatea vizată are stare neconformă",
             "restul verificărilor sunt conforme" if facts["support_ok"] else
             "restul verificărilor sunt neconforme"),
            (f"rubrica urmărită se referă la {FIELD_NAMES[facts['field']]}",
             "rezultatul acestei rubrici este îndeplinit" if facts["field_passed"] else
             "rezultatul acestei rubrici este neîndeplinit",
             "pașii auxiliari sunt păstrați" if facts["support_ok"] else
             "pașii auxiliari nu sunt păstrați"),
        )[style]
        parts = list(clauses)
    random.Random(seed).shuffle(parts)
    intro = ("Registru de parametri imaginați", "Observații despre o structură simulată",
             "Metadate pentru un caz conceptual")[style]
    tail = ("Fără vorbire redată.", "Sunt doar atribute inventate.",
            "Nu reproduce o comunicare.")[style]
    return f"{intro} {symbol}. " + "; ".join(parts) + f". {tail}"


def _pairs(batch_id: str):
    if batch_id not in SPECS:
        raise ValueError("Unknown factor holdout batch")
    split, offset, d1_pairs, s1_pairs = SPECS[batch_id]
    rng = random.Random(918000 + int(batch_id[-4:]))
    for code, count in (("D1", d1_pairs), ("S1", s1_pairs)):
        for index in range(count):
            symbol = f"concept-{batch_id[-4:]}-{index + 1:03d}-{code.lower()}"
            state_rng = random.Random(offset + index * 23 + (0 if code == "D1" else 419))
            kind, negative, positive, judge = BUILDERS[code](index + offset, state_rng, symbol)
            if (set(negative) != set(positive) or
                    sum(negative[key] != positive[key] for key in negative) != 1 or
                    judge(negative) or not judge(positive)):
                raise AssertionError("Invalid one-fact factor holdout pair")
            style = 0 if split == "dev" else 1 if index < count // 2 else 2
            yield code, kind, symbol, negative, positive, rng.randrange(1 << 30), style


def build_d1_facts(batch_id: str) -> list[dict]:
    return [facts for code, _, _, negative, positive, _, _ in _pairs(batch_id)
            if code == "D1" for facts in (negative, positive)]


def build_rows(batch_id: str) -> list[dict]:
    split = SPECS[batch_id][0]
    rows = []
    for code, kind, symbol, negative, positive, seed, style in _pairs(batch_id):
        for variant, facts in enumerate((negative, positive)):
            rows.append({
                "id": f"{batch_id}-ro-{len(rows) + 1:05d}", "language": "ro",
                "source_kind": kind, "split": split,
                "text": render(code, facts, seed, symbol, style),
                "labels": [code] if variant else [],
                "origin": "abstract_factor_holdout_v1",
            })
    return rows


def main() -> None:
    for batch_id, (split, _, _, _) in SPECS.items():
        stem = ROOT / "data/synthetic" / batch_id
        paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
        if any(path.exists() for path in paths):
            raise FileExistsError("Preserve existing factor holdout batch")
        rows = build_rows(batch_id)
        paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                    for row in rows), encoding="utf-8")
        table = [f"# {batch_id} — fișe abstracte pentru factorizarea probelor", "",
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
