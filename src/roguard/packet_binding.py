"""Private exact-byte record for a blinded annotation packet."""

from __future__ import annotations

import json
import stat
from pathlib import Path


def binding_path(answers: Path) -> Path:
    return Path(f"{answers}.packet.json")


def read_binding(answers: Path, expected_digest: str) -> bool:
    """Validate an existing private sidecar; return false only when absent."""
    path = binding_path(answers)
    if not path.exists() and not path.is_symlink():
        return False
    if (path.is_symlink() or not path.is_file() or
            stat.S_IMODE(path.stat().st_mode) & 0o077):
        raise ValueError("Packet binding must be an ordinary private file (mode 0600)")
    binding = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(binding, dict) or set(binding) != {"schema_version", "packet_sha256"} or
            type(binding["schema_version"]) is not int or binding["schema_version"] != 1 or
            binding["packet_sha256"] != expected_digest):
        raise ValueError("Packet binding differs from the assigned packet bytes")
    return True
