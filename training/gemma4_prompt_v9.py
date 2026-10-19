"""Explicit direct-answer templates for Gemma 4 classification."""
from training.evaluate_gemma_review_v9 import completion_prefix
from training.prepare_instruction_sft_v8 import messages


class DirectAnswerTemplate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args, **kwargs):
        # MLX otherwise supplies its own thinking default, unlike the HF
        # tokenizer used by preparation. Bind this behavior explicitly.
        kwargs['enable_thinking'] = False
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def classifier_tokens(tokenizer, row, label_ids):
    return completion_prefix(DirectAnswerTemplate(tokenizer), messages(row), label_ids)
