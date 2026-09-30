"""Optional MLX score head for the pinned Romanian binary adapter.

The returned score is normalized between the next-token logits for lowercase
``da`` and ``nu``. It is a choice score, not a calibrated real-world risk
probability. No text is transmitted or external action is performed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .prompt_v4 import applicable_codes, make_prompt_v4
from .review import CATEGORIES


def choice_score(yes_logit: float, no_logit: float) -> float:
    """Return a numerically stable two-choice softmax score."""
    difference = yes_logit - no_logit
    if difference >= 0:
        return 1.0 / (1.0 + math.exp(-difference))
    exp_value = math.exp(difference)
    return exp_value / (1.0 + exp_value)


def choice_token_ids(tokenizer, messages: list[dict], prompt_ids: list[int]) -> tuple[int, int]:
    """Verify each answer is exactly one next token under the chat template."""
    answer_ids = []
    for answer in ("da", "nu"):
        complete = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}], tokenize=True
        )
        if complete[:len(prompt_ids)] != prompt_ids or len(complete) != len(prompt_ids) + 2:
            raise ValueError("Binary answer is not one token in this chat template")
        if complete[-1] != tokenizer.eos_token_id:
            raise ValueError("Binary answer lacks the expected end token")
        answer_ids.append(complete[len(prompt_ids)])
    if answer_ids[0] == answer_ids[1]:
        raise ValueError("Binary answers tokenize identically")
    return answer_ids[0], answer_ids[1]


class MLXTokenMarginClassifier:
    """Score applicable abstract-card codes from one-step RoMistral logits."""

    def __init__(self, adapter_dir: str | Path):
        import mlx.core as mx
        from mlx_lm import load

        adapter_dir = Path(adapter_dir)
        metadata = json.loads((adapter_dir / "roguard_metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("backend") != "mlx_lm_codes" or metadata.get("prompt_version") != "v4" or
                tuple(metadata.get("categories", ())) != CATEGORIES or metadata.get("language") != "ro"):
            raise ValueError("Token-margin scoring requires a Romanian v4 MLX adapter")
        self.model, self.tokenizer = load(metadata["base_model_path"], adapter_path=str(adapter_dir))
        self.mx = mx

    def score(self, text: str, language: str, source_kind: str) -> dict[str, float]:
        if language != "ro" or not text.strip():
            raise ValueError("Specify nonempty Romanian abstract-card text")
        row = {"language": language, "source_kind": source_kind, "text": text}
        scores = {code: 0.0 for code in CATEGORIES}
        for code in applicable_codes(source_kind):
            messages = [{"role": "user", "content": make_prompt_v4(row, code)}]
            prompt_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            yes_id, no_id = choice_token_ids(self.tokenizer, messages, prompt_ids)
            logits = self.model(self.mx.array([prompt_ids]))[0, -1]
            self.mx.eval(logits)
            scores[code] = choice_score(float(logits[yes_id]), float(logits[no_id]))
        return scores
