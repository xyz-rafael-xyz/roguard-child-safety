"""Pinned Romanian abstract-card NLI questions for the V21 research study."""

from __future__ import annotations

from pathlib import Path

from .prompt_v18 import FIELDS, decision
from .review import sha256

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
MAX_LENGTH = 256
BASE_HASHES = {
    "config.json": "bc7b85f164a17c1b007d87fb99d3676f4a4d2e6511d2288dd3f5334dc0f34e6b",
    "model.safetensors": "7c8e29f1115986d032e92b0fbaa0bdef1062a46f658b08705f237c05014a8541",
    "special_tokens_map.json": "311de3f4eed9d76a43bf0d71f10e62e086ca65ccce9f15d5da0d2098bf519ecc",
    "spm.model": "13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3",
    "tokenizer.json": "e23095eb61ba944c7be3a5d3e8ec19e37ce7ced0daa03550bde03e83c21b3f8a",
    "tokenizer_config.json": "25e9f84c6b43da7bff43a0aee8eeee2c8a39d246823d8464b9275ef704eab42f",
}
HYPOTHESES = {
    "minor_source": "Sursa acestei fișe este un minor.",
    "support_anchor": "În fișă este prezent un reper de siguranță sau de sprijin.",
    "indirect_pattern": "Fișa indică un tipar indirect sau repetat de căutare a sprijinului.",
    "explicit_request": "Fișa indică o cerere explicită de sprijin.",
}
if tuple(HYPOTHESES) != FIELDS:
    raise RuntimeError("V21 hypothesis order differs from D1 factor order")


def verify_base(path: Path) -> None:
    if not path.is_dir() or any(
            not (path / name).is_file() or sha256(path / name) != digest
            for name, digest in BASE_HASHES.items()):
        raise ValueError("NLI base files differ from the pinned revision")


def hypotheses_for(text: str) -> list[tuple[str, str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("V21 requires a nonempty abstract Romanian card")
    return [(text, HYPOTHESES[field]) for field in FIELDS]


def field_probabilities(logits) -> list[float]:
    """Convert entailment/contradiction logits to field scores; neutral is excluded."""
    import torch

    if logits.ndim != 2 or logits.shape[-1] != 3:
        raise ValueError("V21 NLI model must return three class logits")
    return torch.sigmoid((logits[:, 0] - logits[:, 2]).float()).detach().cpu().tolist()
