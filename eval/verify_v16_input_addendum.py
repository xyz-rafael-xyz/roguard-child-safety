"""Verify the pre-author input-integrity layer around the frozen V16 evaluator."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
RECORD = Path("eval/prospective/ro-v16-input-integrity-addendum.json")
BASE_RECORD = Path("eval/prospective/ro-v16-baseline-freeze.json")
FILES = frozenset({
    "src/roguard/annotate_packet_bound.py",
    "src/roguard/human_eval_bound.py",
    "src/roguard/packet_binding.py",
    "src/roguard/review_audit.py",
})
ENTRY_POINTS = {
    "roguard-annotate-packet": "roguard.annotate_packet_bound:main",
    "roguard-human-eval": "roguard.human_eval_bound:main",
}


def verify(root: Path = ROOT) -> dict:
    record = json.loads((root / RECORD).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "registered_input_integrity_extension_before_independent_batch",
        "registered_on": "2026-09-30",
        "pre_extension_git_commit": "3e2a4eecb5d2adbfe6e12c6ecf68ff9a14cc6ca7",
        "base_record_sha256": sha256(root / BASE_RECORD),
        "role": "input_integrity_only_no_scoring_or_target_change",
        "independent_batch_received_at_registration": False,
    }
    if (not isinstance(record, dict) or set(record) != {*expected, "sha256"} or
            any(record.get(name) != value for name, value in expected.items()) or
            date.fromisoformat(record["registered_on"]) > date.today() or
            not isinstance(record["sha256"], dict) or set(record["sha256"]) != FILES):
        raise ValueError("Input-integrity addendum differs")
    for name in FILES:
        if sha256(root / name) != record["sha256"][name]:
            raise ValueError(f"Input-integrity file differs: {name}")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    for name, entry in ENTRY_POINTS.items():
        if not re.search(rf'^{re.escape(name)} = "{re.escape(entry)}"$', pyproject, re.MULTILINE):
            raise ValueError(f"Public CLI entry point differs: {name}")
    return {"status": record["status"], "files_verified": len(FILES),
            "independent_batch_received_at_registration": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
