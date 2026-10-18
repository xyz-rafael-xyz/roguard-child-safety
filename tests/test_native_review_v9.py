from training.train_native_review_v9 import counts, texts


def test_native_counts_keep_absent_language_explicit_without_inventing_rates():
    rows = [dict(language='ro', source='v7_training_frame_projection'), dict(language='ro', source='synthetic')]
    result = counts(rows, [1,0], [.1,.9], {'ro': .5})
    assert result['ro_conversational']['missed'] == 1
    assert result['ro_formal']['false_reviews'] == 1
    assert result['uk_formal']['positive'] == 0
    assert result['uk_formal']['recall'] is None


def test_native_preprocessing_canonicalizes_cedilla_and_excludes_future_turns():
    row = dict(turns=[dict(role='minor',text='Ştiu să citesc.'),dict(role='minor',text='Future material')],focus=0)
    context, focal = texts(row)
    assert focal == 'Știu să citesc.'
    assert 'Future' not in context
