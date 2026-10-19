from training.native_quote_view_v9 import texts
from training.train_bare_distress_v9 import texts as original_texts


def row(text, focus=1):
    return dict(focus=focus, turns=[dict(role='minor', text='Context înainte. Попередній контекст.'),
                                  dict(role='minor', text=text),
                                  dict(role='minor', text='Future turn must not enter the input.')])


def test_full_content_retained_and_attribution_separated():
    sample = row('Nu mi se aplică «un fragment», dar acest lucru mi se aplică.')
    full, scope = texts(sample)
    assert full == original_texts(sample)[0]
    assert 'un fragment' in full and 'un fragment' not in scope
    assert 'nu mi se aplică' in scope.lower()
    assert 'dar acest lucru mi se aplică' in scope.lower()
    assert 'Future turn' not in full


def test_unquoted_and_incomplete_quotations_keep_original_input():
    for text in ('Це мої слова про урок.', 'Nu ascund nimic.', 'Опис «ще не завершено'):
        sample = row(text)
        assert texts(sample) == original_texts(sample)


def test_quote_marker_is_a_feature_not_a_decision_or_content_deletion():
    sample = row('Це справді про мене: «текст». Я підтверджую це.')
    full, scope = texts(sample)
    assert 'текст' in full
    assert 'QUOTED_CONTENT'.lower() in scope.lower()
    assert 'Я підтверджую це'.lower() in scope.lower()


def test_missing_focus_defaults_to_last_turn():
    sample = dict(turns=[dict(role='minor', text='Ştiu că „textul” este citat.')])
    full, scope = texts(sample)
    assert full == original_texts(sample)[0]
    assert 'știu' in scope.lower()
    assert 'textul' not in scope
