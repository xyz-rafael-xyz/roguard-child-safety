"""Reject private research artifacts from the public Git tree and its history."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".ggml", ".pem", ".p12"}
PRIVATE_PATH = re.compile(rb"/Users/" + rb"rafael" + rb"|/home/" + rb"rafael")
CREDENTIAL = re.compile(
    rb"gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    rb"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|"
    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def forbidden(path: str) -> bool:
    candidate = Path(path)
    return (candidate.suffix.lower() in FORBIDDEN_SUFFIXES or
            candidate.name == "adapter_config.json" or
            candidate.name == ".env")


def main() -> None:
    tracked = [item.decode() for item in git("ls-files", "-z").split(b"\0") if item]
    history = [line.decode().split(" ", 1)[1] for line in
               git("rev-list", "--objects", "--all").splitlines() if b" " in line]
    bad = sorted({path for path in tracked + history if forbidden(path)})
    if bad:
        raise SystemExit("Private artifact paths found in public Git history: " + ", ".join(bad))
    for path in tracked:
        data = (ROOT / path).read_bytes()
        if b"\0" in data:
            continue
        if PRIVATE_PATH.search(data) or CREDENTIAL.search(data):
            raise SystemExit(f"Private path or credential-shaped text found in {path}")
    print(f"Public tree audit passed: {len(tracked)} tracked files; no private artifact paths in Git history")


if __name__ == "__main__":
    main()
