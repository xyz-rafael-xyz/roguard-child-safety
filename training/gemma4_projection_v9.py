"""Gemma 4 final-token projection, including the model's logit soft cap."""


def last_logits(model, tokens, cache=None):
    import mlx.core as mx
    lm = getattr(model, 'language_model', model)
    if lm.model_type != 'gemma4_text':
        raise ValueError('Expected Gemma 4 text architecture')
    hidden = lm.model(tokens, cache=cache)[:, -1:, :]
    logits = lm.model.embed_tokens.as_linear(hidden) if lm.tie_word_embeddings else lm.lm_head(hidden)
    cap = lm.final_logit_softcapping
    return logits if cap is None else mx.tanh(logits/cap)*cap
