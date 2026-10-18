"""Gemma text projection helpers; callers verify against full model output."""


def language_model(model):
    return getattr(model, 'language_model', model)


def project(model, hidden):
    lm = language_model(model)
    if lm.model_type != 'gemma3_text':
        raise ValueError('Projection shortcut is restricted to Gemma 3 text')
    return lm.model.embed_tokens.as_linear(hidden) if lm.tie_word_embeddings else lm.lm_head(hidden)


def last_logits(model, tokens, cache=None):
    lm = language_model(model)
    hidden = lm.model(tokens, cache=cache)
    return project(model, hidden[:, -1:, :])
