"""A narrow parser for the pinned Qwen chat template's empty reasoning wrapper."""

from __future__ import annotations

import re

from .prompt_v4 import parse_binary


PARSER_PROFILE = "qwen_empty_think_wrapper_v1"


def parse_qwen_binary(raw: str) -> bool | None:
    """Accept a lone decision, optionally preceded by an empty think block."""

    direct = parse_binary(raw)
    if direct is not None:
        return direct
    match = re.fullmatch(r"<think>\s*</think>\s*(da|nu)", raw.strip())
    return parse_binary(match.group(1)) if match else None
