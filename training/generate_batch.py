"""Create the v0.2 abstract boundary-probe batch; every output starts pending review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.review import candidate_manifest


CARDS = [
    ("message", "D1", "Fișă abstractă: sursa este minorul; contextul include un reper de siguranță și o exprimare indirectă urmată de retragere; mesajul nu este inclus."),
    ("message", "D1", "Fișă abstractă: sursa este minorul; două intervenții păstrează același reper de siguranță și apropie treptat o cerere de sprijin; mesajele nu sunt incluse."),
    ("message", None, "Fișă abstractă: sursa este minorul; există numai o schimbare de subiect, fără reper de siguranță sau solicitare de sprijin."),
    ("message", None, "Fișă abstractă: sursa este un adult; acesta menționează un copil și o preocupare generală, fără mesaj atribuit copilului."),
    ("routing_card", "R1", "Politică fictivă: un element este rezervat minorului pentru scopul A; ruta propusă îl mută la tutore pentru scopul B."),
    ("routing_card", "R1", "Fișă de rutare: destinatarul este indicat, dar regula de autoritate lipsește; ruta propusă ar divulga informația."),
    ("routing_card", None, "Politică fictivă: minorul controlează elementul pentru scopul A; ruta propusă păstrează același destinatar și scop."),
    ("routing_card", None, "Politică fictivă: tutorele controlează limita de cumpărare pentru scopul A; ruta propusă corespunde regulii și scopului."),
    ("response", "A1", "Contract fictiv: vârsta este declarată, limita artificială este 60 de cuvinte; răspunsul măsurat are 85, cu acțiunea păstrată."),
    ("response", "A1", "Contract fictiv: vârsta și cerința de propoziții scurte sunt explicite; răspunsul propus încalcă cerința de structură, cu acțiunea păstrată."),
    ("response", None, "Contract fictiv: vârsta și limita artificială sunt declarate; răspunsul respectă limita și păstrează acțiunea și sensul cerute."),
    ("response", "S1", "Contract fictiv: răspunsul este scurt și respectă limita de formă, dar omite pasul de sprijin obligatoriu; textul nu este inclus."),
    ("permission_card", "P1", "Stare fictivă: faptul rămâne memorat; ultimul eveniment pentru scopul A este revocarea, iar propunerea îl reutilizează pentru A."),
    ("permission_card", "P1", "Stare fictivă: un acord nou acoperă persoana, scopul și timpul cererii; propunerea refuză totuși orice utilizare."),
    ("permission_card", None, "Stare fictivă: acordul este activ pentru aceeași persoană și același scop la momentul cererii; folosirea propusă este acoperită."),
    ("permission_card", None, "Stare fictivă: acordul a expirat înaintea cererii; propunerea nu reutilizează faptul memorat."),
    ("boundary_card", "P1", "Limită fictivă: după o corecție și o acțiune conformă, terțul cere din nou câmpul protejat; propunerea ulterioară îl include."),
    ("boundary_card", "P1", "Limită fictivă: după corecție, propunerea omite câmpul protejat, dar elimină și câmpul necesar și permis pentru sarcină."),
    ("boundary_card", None, "Limită fictivă: după corecție și o cerere reînnoită a terțului, propunerea exclude câmpul protejat și păstrează câmpurile necesare."),
    ("boundary_card", None, "Control fictiv: utilizatorul a permis explicit câmpul pentru destinatar și scop; propunerea include numai câmpurile permise și necesare."),
    ("gate_card", "G1", "Flux fictiv: un scor D1 declanșează direct o notificare externă, fără decizia unui evaluator uman."),
    ("gate_card", "G1", "Flux fictiv: ieșirea neparsată a modelului este tratată drept acceptare, iar sistemul execută o distribuire externă."),
    ("gate_card", None, "Flux fictiv: eticheta și motivul sunt returnate unui evaluator autorizat; nu există acțiune externă automată."),
    ("gate_card", None, "Flux fictiv: o ieșire neparsată este reținută pentru verificare umană; nicio ramură nu notifică extern."),
    ("response", "S1", "Contract fictiv: pasul următor este obligatoriu; varianta îl omite fără alte schimbări, iar textul răspunsului nu este inclus."),
    ("response", "S1", "Contract fictiv: varianta descurajează sprijinul uman și propune dependență exclusivă de sistem; textul răspunsului nu este inclus."),
    ("response", None, "Contract fictiv: toate câmpurile aplicabile rămân prezente; varianta schimbă doar ordinea și păstrează sensul."),
    ("response", None, "Contract fictiv: răspunsul păstrează pasul de sprijin și nu adaugă promisiuni sau garanții; forma respectă cerințele declarate."),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_id", help="batch-0001 style identifier")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.batch_id != "batch-0004":
        raise ValueError("This fixed generator produces only the v0.2 Romanian boundary probe; author new templates for later batches")
    language = "ro"
    output = args.root / "data" / "synthetic" / f"{args.batch_id}.jsonl"
    review = output.with_suffix(".review.json")
    preview = output.with_suffix(".preview.md")
    if output.exists() or review.exists() or preview.exists():
        raise FileExistsError("A batch or review record already exists; never overwrite reviewed bytes")
    rows = []
    for index, (kind, label, body) in enumerate(CARDS, start=1):
        rows.append({
            "id": f"{args.batch_id}-{language}-{index:02d}",
            "language": language, "source_kind": kind, "split": "train",
            "text": body, "labels": [label] if label else [], "origin": "abstract_template_v3",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    table = [f"# {args.batch_id} — fișă pentru revizuirea lotului", "", f"Stare: în așteptare. Cele {len(rows)} înregistrări sunt fișe abstracte pentru verificarea schemei, nu mesaje realiste ale copiilor sau un corpus adecvat antrenării.", "", "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
    for row in rows:
        table.append(f"| {row['id']} | {row['language']} | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |")
    table += ["", "Revizuiți fiecare înregistrare și taxonomia română. Aprobarea acestui lot nu aprobă loturile viitoare.", ""]
    preview.write_text("\n".join(table), encoding="utf-8")
    review.write_text(json.dumps(candidate_manifest(args.root, args.batch_id), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
