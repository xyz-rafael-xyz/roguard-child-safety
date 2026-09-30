"""Bind a local annotation session to the exact blinded packet bytes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from .annotate_packet import _packet_rows, annotate_packet
from .packet_binding import binding_path, read_binding
from .review import sha256


def annotate_packet_bound(
    packet_path: Path,
    answers_path: Path,
    read_line: Callable[[], str],
    show: Callable[[str], None],
) -> dict:
    """Record packet identity before the first answer and check it on resume."""
    packet, answers = packet_path.expanduser().resolve(), answers_path.expanduser()
    if packet == answers.resolve() or (answers.exists() and os.path.samefile(packet, answers)):
        raise ValueError("Packet and answer destination must differ")
    _packet_rows(packet)
    digest = sha256(packet)
    if not read_binding(answers, digest):
        if answers.exists() or answers.is_symlink():
            raise ValueError("Existing answers have no packet binding; start a new private stream")
        answers.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
        descriptor = os.open(binding_path(answers), flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps({"schema_version": 1, "packet_sha256": digest},
                                    sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    try:
        result = annotate_packet(packet, answers, read_line, show)
    finally:
        if sha256(packet) != digest:
            raise ValueError("Blinded packet changed during annotation")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a blinded abstract packet with exact-byte resume binding")
    parser.add_argument("packet", type=Path)
    parser.add_argument("answers", type=Path)
    args = parser.parse_args()
    try:
        result = annotate_packet_bound(
            args.packet, args.answers, sys.stdin.readline,
            lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
