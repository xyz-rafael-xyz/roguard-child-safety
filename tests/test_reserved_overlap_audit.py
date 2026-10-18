from training.audit_reserved_overlap_v9 import keys


def test_input_overlap_ignores_future_and_labels_but_keeps_language_identity():
    row = dict(language='ro', turns=[dict(role='minor', text='Ştiu să citesc.')], focus=0)
    original = keys(row)
    extended = dict(row, turns=row['turns']+[{'invalid': 'unread future'}], attributes={'not': 'used'})
    assert keys(extended) == original
    assert keys(dict(row, language='uk'))[0] != original[0]


def test_cedilla_and_comma_forms_share_overlap_identity_and_context_remains_distinct():
    row = dict(language='ro', turns=[dict(role='minor', text='Ştiu să citesc.')])
    canonical = dict(language='ro', turns=[dict(role='minor', text='Știu să citesc.')])
    assert keys(row) == keys(canonical)
    contextual = dict(language='ro', turns=[dict(role='assistant', text='Context')]+canonical['turns'])
    assert keys(contextual)[0] != keys(canonical)[0]
    assert keys(contextual)[1] == keys(canonical)[1]
