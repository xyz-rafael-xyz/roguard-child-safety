"""Export exact taxonomy bytes for an offline independent review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

from .review import LANGUAGES, taxonomy_sha256
from .v1_review import candidate_digest

ROOT = Path(__file__).resolve().parents[2]
PACKETS = {"ro": "RO_REVIEW_PACKET.md", "uk": "UK_REVIEW_PACKET.md"}
RECORDS = {"ro": "review_ro_independent.json", "uk": "review_uk.json"}
V1_PACKETS = {"ro": "RO_V1_REVIEW_PACKET.md", "uk": "UK_V1_REVIEW_PACKET.md"}
READMES = {
    "ro": ("# Pachet pentru revizuirea taxonomiei\n\n"
           "Citiți mai întâi `docs/RO_REVIEW_PACKET.md`, apoi `taxonomy/taxonomy_ro.md`. "
           "Verificați amprenta din `manifest.json`. Legăturile către materialele private "
           "de cercetare pot să nu fie accesibile; ele oferă doar context suplimentar. "
           "Pachetul nu conține fișe etichetate sau scoruri de model și nu reprezintă "
           "o revizuire finalizată.\n"),
    "uk": ("# Пакет для перегляду таксономії\n\n"
           "Спочатку прочитайте `docs/UK_REVIEW_PACKET.md`, потім "
           "`taxonomy/taxonomy_uk.md`. Звірте контрольну суму в `manifest.json`. "
           "Посилання на приватні дослідницькі матеріали можуть бути недоступними; "
           "вони наведені лише як додатковий контекст. Пакет не містить карток із "
           "мітками або оцінок моделі й не засвідчує завершення перегляду.\n"),
}


def create_review_bundle(root: Path, language: str, output: Path,
                         *, candidate_v1: bool = False) -> dict:
    """Write a create-once private ZIP; this never records or approves a review."""
    if language not in LANGUAGES:
        raise ValueError("Choose Romanian or Ukrainian")
    root = root.resolve()
    requested = output.expanduser().absolute()
    if requested.is_symlink():
        raise ValueError("Review bundle destination must not be a symlink")
    target = requested.resolve()
    if target.suffix != ".zip":
        raise ValueError("Review bundle output must end in .zip")
    if target.is_relative_to(root) and not target.is_relative_to(root / "review_runs"):
        raise ValueError("Reviewer bundle must stay outside Git or under review_runs/")
    if target.exists() or target.is_symlink():
        raise FileExistsError("Reviewer bundle destination already exists")

    taxonomy_name = (f"taxonomy/taxonomy_{language}_v1_candidate.md" if candidate_v1 else
                     f"taxonomy/taxonomy_{language}.md")
    packet_name = V1_PACKETS[language] if candidate_v1 else PACKETS[language]
    names = ((taxonomy_name, f"docs/{packet_name}", "docs/TAXONOMY_REVIEW_V1.md",
              "docs/REVIEW_GATE.md") if candidate_v1 else
             (taxonomy_name, f"docs/{packet_name}", "docs/ROLE_TERMINOLOGY_AUDIT.md",
              "docs/REVIEW_GATE.md"))
    sources = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Review bundle source is missing or linked: {name}")
        sources[name] = path.read_bytes()
    sources["README.md"] = ((
        f"# RoGuard v1 candidate review / Revizuire candidat v1 / Перегляд чернетки v1\n\n"
        f"Read docs/{packet_name} and {taxonomy_name}. Verify manifest.json. "
        "This bundle has no labeled cards or model scores and is not an approval.\n"
    ) if candidate_v1 else READMES[language]).encode("utf-8")
    digest = candidate_digest(root, language) if candidate_v1 else taxonomy_sha256(root, language)
    record_name = f"review_{language}_v1_candidate.json" if candidate_v1 else RECORDS[language]
    record_path = root / "taxonomy" / record_name
    if record_path.is_symlink() or not record_path.is_file():
        raise ValueError("Independent review record is missing or linked")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("status") not in {"pending", "approved"} or
            record.get("taxonomy_sha256") != digest or
            digest.encode("ascii") not in sources[f"docs/{packet_name}"]):
        raise ValueError("Reviewer packet or record differs from the current taxonomy")
    manifest = {
        "schema_version": 1,
        "language": language,
        "taxonomy_sha256": digest,
        **({"taxonomy_version": "v1_candidate"} if candidate_v1 else {}),
        "files_sha256": {name: hashlib.sha256(value).hexdigest()
                         for name, value in sources.items()},
        "review_status_at_export": record["status"],
        "model_scores_or_labeled_cards_included": False,
    }
    sources["manifest.json"] = (json.dumps(manifest, ensure_ascii=False,
                                            sort_keys=True, indent=2) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in sources.items():
                    archive.writestr(name, value)
            stream.flush()
            os.fsync(stream.fileno())
        if (any(hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest["files_sha256"][name]
                for name in names) or
                (candidate_digest(root, language) if candidate_v1 else taxonomy_sha256(root, language)) != digest):
            raise ValueError("Review source changed during export")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return {"output": str(target), "language": language,
            "taxonomy_sha256": digest,
            **({"taxonomy_version": "v1_candidate"} if candidate_v1 else {}),
            "files": list(sources),
            "independent_review_completed_by_export": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=LANGUAGES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--candidate-v1", action="store_true",
                        help="Export the corrected v1 candidate and its pending review record")
    args = parser.parse_args()
    try:
        print(json.dumps(create_review_bundle(args.root, args.language, args.output,
                                              candidate_v1=args.candidate_v1), indent=2))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
