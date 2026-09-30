"""Resumed annotations stay bound to the packet bytes actually displayed."""

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from roguard.annotate_packet_bound import annotate_packet_bound
from roguard.packet_binding import binding_path
from roguard.review import sha256


def _packet(path: Path) -> None:
    rows = [{"item_id": f"item_{i}", "language": "ro", "category": "D1",
             "source_kind": "message", "abstract_card": f"Fișă simbolică {i}."}
            for i in (1, 2)]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


class PacketBindingTests(unittest.TestCase):
    def test_resume_requires_original_packet_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, answers = Path(directory) / "packet.jsonl", Path(directory) / "answers.jsonl"
            _packet(packet)
            shown = []
            entries = iter(["1\n", "1\n", "2\n", "1\n"])
            with self.assertRaisesRegex(ValueError, "incompletă"):
                annotate_packet_bound(packet, answers, lambda: next(entries, ""), shown.append)
            binding = binding_path(answers)
            self.assertEqual(stat.S_IMODE(binding.stat().st_mode), 0o600)
            self.assertEqual(json.loads(binding.read_text())["packet_sha256"], sha256(packet))
            self.assertEqual(len(answers.read_text().splitlines()), 1)
            altered = [json.loads(line) for line in packet.read_text().splitlines()]
            altered[1]["abstract_card"] = "Altă fișă simbolică."
            packet.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                      for row in altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding differs"):
                annotate_packet_bound(packet, answers, lambda: "", shown.append)
            self.assertEqual(len(answers.read_text().splitlines()), 1)

    def test_missing_or_public_binding_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, answers = Path(directory) / "packet.jsonl", Path(directory) / "answers.jsonl"
            _packet(packet)
            answers.write_text("", encoding="utf-8")
            os.chmod(answers, 0o600)
            with self.assertRaisesRegex(ValueError, "no packet binding"):
                annotate_packet_bound(packet, answers, lambda: "", lambda _: None)
            binding = binding_path(answers)
            binding.write_text(json.dumps({"schema_version": 1,
                                           "packet_sha256": sha256(packet)}))
            with self.assertRaisesRegex(ValueError, "private file"):
                annotate_packet_bound(packet, answers, lambda: "", lambda _: None)

    def test_packet_change_while_displayed_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, answers = Path(directory) / "packet.jsonl", Path(directory) / "answers.jsonl"
            _packet(packet)
            entries = iter(["1\n", "1\n", "2\n", "1\n", "1\n", "1\n", "2\n", "1\n"])
            def show(message: str) -> None:
                if message.startswith("[2/2]"):
                    packet.write_text(packet.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed during annotation"):
                annotate_packet_bound(packet, answers, lambda: next(entries, ""), show)


if __name__ == "__main__":
    unittest.main()
