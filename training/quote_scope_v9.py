"""Research text view separating closed quotations from their surrounding scope.

No language-specific decision rules or benchmark phrases are used. Unclosed
quotations remain intact. This transform is not itself a safety decision.
"""
import re
import unicodedata

QUOTATIONS = re.compile(r'«[^«»]*»|„[^„“”]*[“”]|“[^“”]*”|"[^"\n]*"')


def abstract_quotes(text):
    text = unicodedata.normalize('NFC', text)
    return QUOTATIONS.sub(' QUOTED_CONTENT ', text)


def scope_text(row, focus):
    turns = row['turns'][:row['focus']+1]
    return abstract_quotes(turns[-1]['text']) if focus else '\n'.join(
        t['role']+': '+abstract_quotes(t['text']) for t in turns[:-1])
