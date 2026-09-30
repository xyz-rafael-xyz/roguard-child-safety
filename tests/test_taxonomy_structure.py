"""Keep both taxonomy drafts in the requested compact, reviewable shape."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODES = ("D1", "R1", "A1", "P1", "G1", "S1")
FIELDS = {
    "ro": ("Definiție", "În sfera categoriei", "Explicit în afara sferei",
           "Benzi de încredere și gravitate", "Decizia de rutare", "Exemple ilustrative"),
    "uk": ("Визначення", "У межах категорії", "Поза межами",
           "Рівні впевненості та пріоритету", "Рішення маршрутизації",
           "Ілюстративні приклади"),
}
HEADER = re.compile(r"^### (D1|R1|A1|P1|G1|S1) — (.+)$", re.MULTILINE)
WORD = re.compile(r"[\w]+(?:[’'-][\w]+)*", re.UNICODE)


class TaxonomyStructureTests(unittest.TestCase):
    def test_required_fields_order_length_and_examples(self):
        for language in ("ro", "uk"):
            text = (ROOT / f"taxonomy/taxonomy_{language}.md").read_text(encoding="utf-8")
            headings = list(HEADER.finditer(text))
            self.assertEqual([match.group(1) for match in headings], list(CODES))
            for index, match in enumerate(headings):
                with self.subTest(language=language, category=match.group(1)):
                    end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
                    section = text[match.start():end].split("\n## ", 1)[0]
                    fields = re.findall(r"^\*\*(.+?):\*\*", section, re.MULTILINE)
                    self.assertEqual(fields, list(FIELDS[language]))
                    self.assertTrue(match.group(2).strip())
                    self.assertGreaterEqual(len(WORD.findall(section)), 200)
                    self.assertLessEqual(len(WORD.findall(section)), 400)
                    examples = section.split(f"**{FIELDS[language][-1]}:**", 1)[1]
                    self.assertEqual(examples.count("(1)"), 1)
                    self.assertEqual(examples.count("(2)"), 1)


if __name__ == "__main__":
    unittest.main()
