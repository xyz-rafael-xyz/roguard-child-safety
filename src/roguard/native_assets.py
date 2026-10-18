"""Reject unregistered files that can override a pinned native tokenizer/model."""
from pathlib import Path


TOKENIZER_FILES = frozenset({
    'config.json', 'tokenizer_config.json', 'special_tokens_map.json',
    'tokenizer.json', 'added_tokens.json', 'vocab.txt', 'vocab.json', 'merges.txt',
    'tokenizer.model', 'sentencepiece.bpe.model', 'chat_template.jinja',
    'chat_template.json',
})


def reject_native_overrides(base, registered_names):
    base = Path(base)
    registered = set(registered_names)
    if any(not isinstance(name, str) or not name or Path(name).name != name
           or name in ('.', '..') or '\\' in name for name in registered):
        raise ValueError('Native asset names must be plain filenames')
    if (base/'adapter_config.json').exists():
        raise ValueError('Unexpected adapter configuration in native base')
    for name in TOKENIZER_FILES - registered:
        if (base/name).exists():
            raise ValueError('Unregistered native tokenizer file: '+name)
