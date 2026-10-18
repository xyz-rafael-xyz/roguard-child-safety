"""Reload the experimental quotation-scope gate without executable objects."""
import numpy as np
from scipy.sparse import hstack
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from training.quote_scope_v9 import scope_text


def score(spec,rows,baseline):
    if spec['format']!='roguard-scope-gate-experiment-v1':raise ValueError('Unexpected format')
    if len(rows)!=len(baseline):raise ValueError('Baseline population differs')
    blocks=[]
    for part in spec['vectorizers']:
        vectorizer=TfidfVectorizer(analyzer=part['analyzer'],ngram_range=tuple(part['ngram_range']),
            vocabulary=part['vocabulary'],sublinear_tf=True)
        vectorizer.idf_=np.asarray(part['idf'])
        blocks.append(vectorizer.transform([scope_text(r,part['focus']) for r in rows]))
    values=expit(hstack(blocks).tocsr()@np.asarray(spec['weights'])+spec['intercept'])
    return np.asarray(baseline)*values**spec['selected']['power']
