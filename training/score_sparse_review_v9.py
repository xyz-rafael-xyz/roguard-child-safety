"""Reconstruct research lexical heads from their non-executable JSON export."""
import numpy as np
from scipy.sparse import hstack
from scipy.special import expit, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from training.fit_sparse_review_v9 import texts
from training.policy_probability_v9 import review_mass


def score(spec, rows):
    if spec['format'] != 'roguard-sparse-review-experiment-v1':
        raise ValueError('Unexpected research artifact format')
    blocks = []
    for block in spec['vectorizers']:
        v = TfidfVectorizer(analyzer=block['analyzer'], ngram_range=tuple(block['ngram_range']),
                            sublinear_tf=True, vocabulary=block['vocabulary'])
        v.idf_ = np.asarray(block['idf'])
        blocks.append(v.transform([texts(r, block['focus']) for r in rows]))
    matrix = hstack(blocks).tocsr()
    probabilities = {}
    for task, head in spec['heads'].items():
        logits = np.asarray(matrix @ np.asarray(head['weights']).T) + np.asarray(head['intercept'])
        if logits.shape[1] == 1:
            p = expit(logits[:, 0])
            p = np.column_stack((1-p, p))
        else:
            p = softmax(logits, axis=1)
        probabilities[task] = {label:p[:,i] for i,label in enumerate(head['classes'])}
    if spec['selected']['mode'] == 'direct':
        return probabilities['review']['1']
    return np.clip(review_mass(probabilities), 0, 1)
