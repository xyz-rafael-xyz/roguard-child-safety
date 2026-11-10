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
    "src/roguard/v10_models/ro-binary.npz": "318e835a26645449fe99841854b0f40b5b5801b3400eae812e545db225dc99c0",
    "src/roguard/v10_models/ro-concern.npz": "7dc016cc4fcf652c62d0122388f8844701e2132f2a400af81599f3ea88a96d90",
    "src/roguard/v10_models/uk-binary.npz": "a4f5aec644793e8e39b55a3b5bfb9a810ebfff660c2d36b60e4f82436a638529",
    "src/roguard/v10_models/uk-concern.npz": "8e1e3c1431c0db78eb6652bdc737ec9e4dc3adf01adb0265943295c28eae2361",
}
PUBLIC_MODEL_HEAD_HISTORY = {
    "src/roguard/v10_models/ro-binary.npz": {
        "2330dc230f5554caac1ae30591adba4d0294f32866df42edccf863996c11f195",
        "b6ca7f6b523a8a52aa0f9cd4a15b9675172127774544c725546ae1a4b8f0867a",
        "99ad81ebdf9aec5a26b4f4c31fe3435d5762d132d3c8036318da15d35ac33c83",
    },
    "src/roguard/v10_models/ro-concern.npz": {
        "38421e16cd6203ed467165bec378c88a20d19320dcbcee063b480444e4373eab",
        "7eb43d50b7eda25045eb7d166e58cb9d3316fa3cdc09044f1572a00f4e776711",
        "c11ea69cd8c2f6e1c44c740cc781105930ff39dd77c0a4888fa6381fd5060066",
    },
    "src/roguard/v10_models/uk-binary.npz": {
        "bded560d29b340a146042487e4b2a55aa9aeb63a49c703c58840973c5c33d00c",
        "59691ae37811ac0a7e247e7a8acaf7fa11aaf6ca5c030710479f1f3ffa60974d",
        "c78604cd0502afd6318a008fb3762e7ecd24df845684db051f48e77e3bcddefe",
    },
    "src/roguard/v10_models/uk-concern.npz": {
        "7cbf091e31e435e555cfa7789982e4060592c02e158294809cc08fd95ba806d7",
        "991019c190bce11cdc45715f314c2ec81a6771d78feab7b61c14bfa94843f101",
        "28133c64217861494d18292ea2163c1527fd117f077e83f76c494575743cc802",
    },
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
    sparse = {line[2:].decode() for line in git("ls-files", "-v", "-z").split(b"\0")
              if line.startswith(b"S ")}
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
        if (path in PUBLIC_MODEL_HEADS and
                hashlib.sha256(git("cat-file", "blob", oid)).hexdigest() not in
                PUBLIC_MODEL_HEAD_HISTORY[path] | {PUBLIC_MODEL_HEADS[path]}):
            raise SystemExit("Public v10 model head changed in Git history: " + path)
    for path in tracked:
        local = ROOT / path
        if not local.is_file():
            if path in sparse:
                continue  # CI's full checkout audits these paths; sparse local worktrees omit them.
            raise SystemExit("Tracked public file missing: " + path)
        data = local.read_bytes()
        if b"\0" in data:
            continue
        if PRIVATE_PATH.search(data) or CREDENTIAL.search(data):
            raise SystemExit(f"Private path or credential-shaped text found in {path}")
    print(f"Public tree audit passed: {len(tracked)} tracked files; no private artifact paths in Git history")


if __name__ == "__main__":
    main()
