"""Standard-library inference for the frozen sparse quotation-scope artifact.

Implements the exported lowercase, char/word n-gram, sublinear TF, IDF and
per-block L2 normalization. It never deserializes executable Python objects.
"""
from collections import Counter
import math
from numbers import Real
import re

from training.quote_scope_v9 import scope_text

_WORDS = re.compile(r'(?u)\b\w\w+\b')
_WHITESPACE = re.compile(r'\s\s+')


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric parameter')
    return float(value)


class ScopeGate:
    def __init__(self, spec):
        if spec.get('format') != 'roguard-scope-gate-experiment-v1':
            raise ValueError('Unsupported scope artifact format')
        self.intercept = _number(spec['intercept'])
        self.power = spec['selected']['power']
        if type(self.power) is not int or self.power not in (1, 2, 4):
            raise ValueError('Unsupported scope exponent')
        weights = tuple(_number(v) for v in spec['weights'])
        self.blocks = []
        offset = 0
        for part in spec['vectorizers']:
            analyzer = part['analyzer']
            if analyzer not in ('char', 'word') or type(part['focus']) is not bool:
                raise ValueError('Unsupported feature view')
            low, high = part['ngram_range']
            if type(low) is not int or type(high) is not int or not 1 <= low <= high <= 5:
                raise ValueError('Invalid n-gram range')
            vocabulary = part['vocabulary']
            size = len(vocabulary)
            if (not size or any(not isinstance(k, str) or type(v) is not int for k, v in vocabulary.items())
                    or sorted(vocabulary.values()) != list(range(size))):
                raise ValueError('Vocabulary indices must be contiguous and unique')
            idf = tuple(_number(v) for v in part['idf'])
            if len(idf) != size or any(v <= 0 for v in idf) or offset+size > len(weights):
                raise ValueError('Invalid feature dimensions or IDF')
            self.blocks.append((part['focus'], analyzer, low, high, dict(vocabulary), idf, weights[offset:offset+size]))
            offset += size
        if not self.blocks or offset != len(weights):
            raise ValueError('Weight dimensions differ')

    def probability(self, row):
        contributions = []
        for focus, analyzer, low, high, vocabulary, idf, weights in self.blocks:
            text = scope_text(row, focus).lower()
            if analyzer == 'char':
                units = _WHITESPACE.sub(' ', text)
                grams = (units[i:i+n] for n in range(low, high+1) for i in range(len(units)-n+1))
            else:
                units = _WORDS.findall(text)
                grams = (' '.join(units[i:i+n]) for n in range(low, high+1) for i in range(len(units)-n+1))
            counts = Counter(vocabulary[g] for g in grams if g in vocabulary)
            values = [(index, (1+math.log(count))*idf[index]) for index, count in counts.items()]
            norm = math.sqrt(math.fsum(value*value for _, value in values))
            contributions.append(math.fsum(weights[index]*value for index, value in values)/norm if norm else 0.)
        margin = self.intercept+math.fsum(contributions)
        if margin >= 0:
            return 1/(1+math.exp(-margin))
        exp = math.exp(margin)
        return exp/(1+exp)

    def score(self, rows, baseline):
        if len(rows) != len(baseline):
            raise ValueError('Baseline population differs')
        result = []
        for row, base in zip(rows, baseline):
            base = _number(base)
            if not 0 <= base <= 1:
                raise ValueError('Baseline score outside [0,1]')
            result.append(base*self.probability(row)**self.power)
        return result
