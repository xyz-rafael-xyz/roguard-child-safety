"""Two native-encoder views: complete content and quotation-abstracted focus.

The complete conversation remains available, including all quoted content.
The second view isolates surrounding attribution without making a decision.
"""
from roguard.semantic import normalize_semantic_text
from training.quote_scope_v9 import abstract_quotes
from training.train_bare_distress_v9 import texts as original_texts

PREPROCESSING = 'native-full-context-quoted-focus-abstraction-v1'


def texts(row):
    full, _ = original_texts(row)
    focus = row.get('focus', len(row['turns'])-1)
    abstracted = abstract_quotes(row['turns'][focus]['text'])
    abstracted = normalize_semantic_text(abstracted).translate(str.maketrans('şţŞŢ', 'șțȘȚ'))
    return full, abstracted
