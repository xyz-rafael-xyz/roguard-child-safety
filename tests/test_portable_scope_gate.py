import copy

import pytest

from training.portable_scope_gate_v9 import ScopeGate


def minimal():
    return dict(format='roguard-scope-gate-experiment-v1', intercept=0., weights=[1.], selected=dict(power=1),
        vectorizers=[dict(focus=True, analyzer='word', ngram_range=[1, 1], vocabulary={'salut': 0}, idf=[1.])])


def test_portable_scope_matches_fitted_multilingual_vectorizer():
    np = pytest.importorskip('numpy')
    pytest.importorskip('sklearn')
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from training.quote_scope_v9 import scope_text
    from training.score_scope_gate_v9 import score
    texts = ['Nu e povestea mea: «un cântec».', 'Це мій текст: «пісня».',
             'Eu povestesc. Eu povestesc.', 'Прочитай		вірш, будь ласка.',
             'Și țin minte acest fragment.', 'Ми читаємо новий уривок.']
    rows = [dict(focus=1, turns=[dict(role='minor', text='Context comun Спільний контекст'),
                               dict(role='minor', text=text)]) for text in texts]
    blocks = []
    specs = []
    for focus, analyzer, ngrams in ((True, 'char', (2, 5)), (True, 'word', (1, 3)), (False, 'word', (1, 2))):
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, sublinear_tf=True)
        blocks.append(vec.fit_transform([scope_text(r, focus) for r in rows]))
        specs.append(dict(focus=focus, analyzer=analyzer, ngram_range=ngrams,
                          vocabulary=vec.vocabulary_, idf=vec.idf_.tolist()))
    head = LogisticRegression(C=2).fit(hstack(blocks), [0, 1, 1, 0, 1, 0])
    spec = dict(format='roguard-scope-gate-experiment-v1', vectorizers=specs, weights=head.coef_[0].tolist(),
                intercept=float(head.intercept_[0]), selected=dict(power=2))
    unseen = copy.deepcopy(rows)
    unseen[0]['turns'][1]['text'] = 'Și acum citesc „un alt fragment”. Citesc, citesc.'
    unseen[1]['turns'][0]['text'] = ''
    unseen[2]['turns'][1]['text'] = '     '
    baseline = np.linspace(.1, .9, len(rows))
    expected = score(spec, unseen, baseline)
    assert np.max(np.abs(ScopeGate(spec).score(unseen, baseline)-expected)) < 1e-12


@pytest.mark.parametrize('change', ['weights', 'idf', 'indices', 'power', 'format'])
def test_malformed_numeric_artifacts_are_rejected(change):
    spec = minimal()
    if change == 'weights':
        spec['weights'] = [float('nan')]
    elif change == 'idf':
        spec['vectorizers'][0]['idf'] = [0.]
    elif change == 'indices':
        spec['vectorizers'][0]['vocabulary'] = {'salut': 1}
    elif change == 'power':
        spec['selected']['power'] = True
    else:
        spec['format'] = 'unknown'
    with pytest.raises(ValueError):
        ScopeGate(spec)
