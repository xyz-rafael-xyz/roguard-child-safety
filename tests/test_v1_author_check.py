import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

from roguard.v1_author_check import check_v1_author_workbook
from roguard.v1_study_kit import create_v1_author_kit


class V1AuthorCheckTests(unittest.TestCase):
    def test_completed_workbook_must_match_issued_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            taxonomy_name = "taxonomy/taxonomy_ro_v1_candidate.md"
            taxonomy = "Taxonomie abstractă pentru verificare.\n".encode("utf-8")
            digest = hashlib.sha256(Path(taxonomy_name).name.encode() + b"\0" + taxonomy + b"\0").hexdigest()
            frozen = {"path": "eval/prospective/freeze.json", "sha256": "a" * 64,
                      "candidate_taxonomy_sha256": digest}
            kits = {category: root / f"review_runs/kit-{category}"
                    for category in ("D1", "S1")}
            with patch("roguard.v1_study_kit.verify_v1_study_freeze", return_value=frozen):
                for category, kit in kits.items():
                    create_v1_author_kit(root, "ro", category, Path(frozen["path"]), kit)
            role = "RO-A1"
            packet = root / f"{role}-v1-author-packet.zip"
            entries = {"START-HERE.txt": b"Abstract study instructions.\n",
                       taxonomy_name: taxonomy}
            for category in ("D1", "S1"):
                for extension in ("json", "txt"):
                    name = f"{role}.{extension}"
                    path = kits[category] / name
                    entries[f"{category}/{name}"] = path.read_bytes()
            with ZipFile(packet, "w") as archive:
                for name, content in entries.items():
                    info = ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0))
                    archive.writestr(info, content)
            workbook = root / "completed.json"
            completed = json.loads((kits["D1"] / f"{role}.json").read_text(encoding="utf-8"))
            completed["native_language_confirmed_by_author"] = True
            completed["independent_of_model_work_confirmed_by_author"] = True
            completed["authored_at"] = "2026-10-04"
            for pair in completed["pairs"]:
                number = str(pair["pair_index"])
                pair["negative_abstract_card"] = "Descriere abstractă a câmpului fictiv " + number + " fără atribut"
                pair["positive_abstract_card"] = "Descriere abstractă a câmpului fictiv " + number + " cu atribut"
            workbook.write_text(json.dumps(completed, ensure_ascii=False), encoding="utf-8")
            result = check_v1_author_workbook(packet, workbook)
            self.assertTrue(result["workbook_compatible"])
            self.assertEqual(result["cards"], 48)
            self.assertFalse(result["card_text_in_report"])

            completed["pairs"][0]["factor"] = "unissued_factor"
            workbook.write_text(json.dumps(completed, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Pair assignment differs"):
                check_v1_author_workbook(packet, workbook)

            workbook.write_text(json.dumps({"metadata": {}, "test_pairs": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "role, language, category"):
                check_v1_author_workbook(packet, workbook)


if __name__ == "__main__":
    unittest.main()
