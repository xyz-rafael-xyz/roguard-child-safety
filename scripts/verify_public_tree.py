"""Reject private research artifacts from the public Git tree and its history."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".safetensors", ".npz", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".ggml", ".pem", ".p12"}
# These four frozen heads are intentionally public release assets. Their exact
# bytes are pinned here; arbitrary research tensors remain forbidden.
PUBLIC_MODEL_HEADS = {
    "src/roguard/v10_models/ro-binary.npz": "2330dc230f5554caac1ae30591adba4d0294f32866df42edccf863996c11f195",
    "src/roguard/v10_models/ro-concern.npz": "38421e16cd6203ed467165bec378c88a20d19320dcbcee063b480444e4373eab",
    "src/roguard/v10_models/uk-binary.npz": "bded560d29b340a146042487e4b2a55aa9aeb63a49c703c58840973c5c33d00c",
    "src/roguard/v10_models/uk-concern.npz": "7cbf091e31e435e555cfa7789982e4060592c02e158294809cc08fd95ba806d7",
}
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
    return (candidate.suffix.lower() in FORBIDDEN_SUFFIXES and path not in PUBLIC_MODEL_HEADS or
            candidate.name == "adapter_config.json" or
            candidate.name == ".env")


def main() -> None:
    tracked = [item.decode() for item in git("ls-files", "-z").split(b"\0") if item]
    history = [line.decode().split(" ", 1) for line in
               git("rev-list", "--objects", "--all").splitlines() if b" " in line]
    bad = sorted({path for path in tracked + [path for _, path in history] if forbidden(path)})
    if bad:
        raise SystemExit("Private artifact paths found in public Git history: " + ", ".join(bad))
    if not set(PUBLIC_MODEL_HEADS) <= set(tracked):
        raise SystemExit("Public v10 model heads missing from the release tree")
    for path, expected in PUBLIC_MODEL_HEADS.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise SystemExit("Public v10 model head differs: " + path)
    for oid, path in history:
        if path in PUBLIC_MODEL_HEADS and hashlib.sha256(git("cat-file", "blob", oid)).hexdigest() != PUBLIC_MODEL_HEADS[path]:
            raise SystemExit("Public v10 model head changed in Git history: " + path)
    for path in tracked:
        data = (ROOT / path).read_bytes()
        if b"\0" in data:
            continue
        if PRIVATE_PATH.search(data) or CREDENTIAL.search(data):
            raise SystemExit(f"Private path or credential-shaped text found in {path}")
    print(f"Public tree audit passed: {len(tracked)} tracked files; no private artifact paths in Git history")


if __name__ == "__main__":
    main()
