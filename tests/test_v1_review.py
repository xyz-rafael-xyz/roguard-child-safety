import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from roguard.v1_review import audit_candidate_reviews, candidate_digest

ROOT = Path(__file__).resolve().parents[1]
CODES = ("D1", "R1", "A1", "P1", "G1", "S1")
WORD = re.compile(r"[\w]+(?:[’'-][\w]+)*", re.UNICODE)
FIELDS = {
    "ro": ("Definiție", "În sfera categoriei", "Explicit în afara sferei",
           "Susținerea etichetei și prioritatea revizuirii", "Decizia de rutare", "Exemple ilustrative"),
    "uk": ("Визначення", "У межах категорії", "Поза межами",
           "Рівень підтвердження й пріоритет перегляду", "Рішення маршрутизації",
           "Ілюстративні приклади"),
}


class V1ReviewTests(unittest.TestCase):
    def test_candidate_shape_and_owner_attested_review_gate(self):
        report = audit_candidate_reviews(ROOT)
        self.assertTrue(report["machine_taxonomy_gate_passed"])
        self.assertFalse(report["reviewer_identity_and_independence_verified_by_software"])
        for language in ("ro", "uk"):
            self.assertEqual(report["taxonomy"][language]["status"], "record_structure_verified")
            text = (ROOT / f"taxonomy/taxonomy_{language}_v1_candidate.md").read_text(encoding="utf-8")
            headings = list(re.finditer(r"^### (D1|R1|A1|P1|G1|S1) — ", text, re.M))
            self.assertEqual([match.group(1) for match in headings], list(CODES))
            for index, heading in enumerate(headings):
                end = headings[index + 1].start() if index + 1 < len(headings) else text.index("\n## ", heading.start())
                section = text[heading.start():end]
                self.assertEqual(re.findall(r"^\*\*(.+?):\*\*", section, re.M), list(FIELDS[language]))
                self.assertTrue(200 <= len(WORD.findall(section)) <= 400)
                self.assertIn("(1)", section)
                self.assertIn("(2)", section)

    def test_changed_candidate_bytes_invalidate_review_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "taxonomy").mkdir()
            for language in ("ro", "uk"):
                for name in (f"taxonomy_{language}_v1_candidate.md", f"review_{language}_v1_candidate.json"):
                    shutil.copyfile(ROOT / "taxonomy" / name, root / "taxonomy" / name)
            path = root / "taxonomy/taxonomy_ro_v1_candidate.md"
            path.write_text(path.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
            report = audit_candidate_reviews(root)
            self.assertEqual(report["taxonomy"]["ro"]["status"], "invalid_digest")
            self.assertEqual(report["taxonomy"]["uk"]["status"], "record_structure_verified")

    def test_same_reviewer_id_across_languages_fails_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "taxonomy").mkdir()
            for language in ("ro", "uk"):
                for name in (f"taxonomy_{language}_v1_candidate.md", f"review_{language}_v1_candidate.json"):
                    shutil.copyfile(ROOT / "taxonomy" / name, root / "taxonomy" / name)
            uk = root / "taxonomy/review_uk_v1_candidate.json"
            record = json.loads(uk.read_text(encoding="utf-8"))
            record["reviews"][0]["reviewer"] = "RO-L-20261004"
            uk.write_text(json.dumps(record), encoding="utf-8")
            report = audit_candidate_reviews(root)
            self.assertFalse(report["machine_taxonomy_gate_passed"])
            self.assertEqual(report["taxonomy"]["uk"]["status"], "duplicate_reviewer_id")


if __name__ == "__main__":
    unittest.main()
