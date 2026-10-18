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


def test_new_focus_families_preserve_the_current_frame_and_distinguish_task_difficulty():
    from training.focus_boundary_contrasts_v9 import generate
    from training.prepare_instruction_sft_v8 import target
    from training.review_data_v8 import input_key
    rows = generate()
    assert len({input_key(r) for r in rows}) == len(rows)
    assert {r['language'] for r in rows} == {'ro','uk'}
    assert any('veselă' in t['text'] for r in rows for t in r['turns'])
    assert any('веселою' in t['text'] for r in rows for t in r['turns'])
    for row in rows:
        assert row['focus'] == len(row['turns'])-1
        assert row['fictional'] is True
        if row['attributes']['concern'] == '0':
            assert target(row) == 0
        elif row['variant'] in ('fiction_after_real','clarify_fictional'):
            assert target(row) == 0
        else:
            assert target(row) == 1


def test_time_variants_preserve_review_target_and_training_family():
    from collections import defaultdict
    from training.focus_boundary_contrasts_v9 import generate as focus
    from training.time_scope_contrasts_v9 import generate
    from training.prepare_instruction_sft_v8 import target
    from training.review_data_v8 import input_key
    rows = generate()
    assert len({input_key(r) for r in rows}) == len(rows)
    families = {r['family'] for r in focus()}
    times = defaultdict(set)
    for row in rows:
        assert row['family'] in families
        assert target(row) == int(row['attributes']['concern'])
        assert row['turns'][row['focus']]['role'] == 'minor'
        times[row['family']].add(row['attributes']['time'])
    assert all(values == {'CURRENT', 'PAST'} for values in times.values())
