"""New abstract Romanian D1 surfaces for a joint four-field encoder study."""

from __future__ import annotations

import argparse
import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILD_D1 = REFERENCE["BUILDERS"]["D1"]
SPECS = {"batch-0035": ("dev", 24, 1901), "batch-0036": ("test", 48, 2101)}


def render(facts: dict, seed: int, symbol: str, style: int) -> str:
    if style == 0:
        clauses = [
            "rolul emițătorului este minor" if facts["role"] == "minor" else
            "rolul emițătorului este adult",
            "rubrica de protecție este activată" if facts["anchor"] else
            "rubrica de protecție este dezactivată",
            "revenirea ocolită spre sprijin este înscrisă" if facts["pattern"] else
            "revenirea ocolită spre sprijin este exclusă",
            "cererea directă de sprijin este înscrisă" if facts["explicit"] else
            "cererea directă de sprijin este exclusă",
        ]
        intro, tail = "Tabel ipotetic de atribute", "Nu este redată o comunicare."
    elif style == 1:
        clauses = [
            "proveniența indică un minor" if facts["role"] == "minor" else
            "proveniența indică un adult",
            "semnul contextual de siguranță se păstrează" if facts["anchor"] else
            "semnul contextual de siguranță se elimină",
            "căutarea indirectă sau repetată a ajutorului se păstrează" if facts["pattern"] else
            "căutarea indirectă sau repetată a ajutorului se elimină",
            "solicitarea deschisă de sprijin se păstrează" if facts["explicit"] else
            "solicitarea deschisă de sprijin se elimină",
        ]
        intro, tail = "Model conceptual de verificare", "Conține numai stări create pentru test."
    elif style == 2:
        clauses = [
            "sursa are statut de minor" if facts["role"] == "minor" else
            "sursa are statut de adult",
            "există reper privind protecția" if facts["anchor"] else
            "nu există reper privind protecția",
            "există un traseu reluat ori indirect către ajutor" if facts["pattern"] else
            "nu există traseu reluat ori indirect către ajutor",
            "există o cerere explicită de sprijin" if facts["explicit"] else
            "nu există cerere explicită de sprijin",
        ]
        intro, tail = "Schemă artificială de observații", "Fără dialog sau mesaj redat."
    else:
        raise ValueError("Unknown holdout surface")
    random.Random(seed).shuffle(clauses)
    return f"{intro} {symbol}. " + "; ".join(clauses) + f". {tail}"


def build_d1_facts(batch_id: str) -> list[dict]:
    return [facts for negative, positive, _, _ in pairs(batch_id)
            for facts in (negative, positive)]


def pairs(batch_id: str):
    if batch_id not in SPECS:
        raise ValueError("Unknown joint-factor holdout batch")
    split, pair_count, offset = SPECS[batch_id]
    rng = random.Random(735000 + int(batch_id[-4:]))
    for index in range(pair_count):
        symbol = f"schemă-{batch_id[-4:]}-{index + 1:03d}"
        state_rng = random.Random(offset + index * 29)
        kind, negative, positive, judge = BUILD_D1(index + offset, state_rng, symbol)
        if (kind != "message" or set(negative) != set(positive) or
                sum(negative[key] != positive[key] for key in negative) != 1 or
                judge(negative) or not judge(positive)):
            raise AssertionError("Invalid one-fact joint D1 holdout pair")
        style = 0 if split == "dev" else 1 if index < pair_count // 2 else 2
        yield negative, positive, rng.randrange(1 << 30), style


def build_rows(batch_id: str) -> list[dict]:
    split = SPECS[batch_id][0]
    rows = []
    for index, (negative, positive, seed, style) in enumerate(pairs(batch_id), 1):
        symbol = f"schemă-{batch_id[-4:]}-{index:03d}"
        for variant, facts in enumerate((negative, positive)):
            rows.append({
                "id": f"{batch_id}-ro-{len(rows) + 1:05d}", "language": "ro",
                "source_kind": "message", "split": split,
                "text": render(facts, seed, symbol, style),
                "labels": ["D1"] if variant else [],
                "origin": "abstract_joint_factor_holdout_v1",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=tuple(SPECS), required=True)
    args = parser.parse_args()
    for batch_id in (args.batch,):
        split, pair_count, _ = SPECS[batch_id]
        stem = ROOT / "data/synthetic" / batch_id
        paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
        if any(path.exists() for path in paths):
            raise FileExistsError("Preserve existing joint-factor holdout batch")
        rows = build_rows(batch_id)
        paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                    for row in rows), encoding="utf-8")
        table = [f"# {batch_id} — fișe abstracte pentru clasificarea comună a atributelor", "",
                 f"Stare: în așteptarea validării automate. Split: `{split}`. {len(rows)} fișe simbolice fără comunicări reale ori realiste.", "",
                 "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
        table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                     for row in rows)
        table.extend(("", "Perechile adiacente au o singură diferență verificată prin regula tipizată.", ""))
        paths[1].write_text("\n".join(table), encoding="utf-8")
        paths[2].write_text(json.dumps(candidate_manifest(ROOT, batch_id), ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"{batch_id}: {pair_count} pairs, {len(rows)} {split} cards")


if __name__ == "__main__":
    main()
