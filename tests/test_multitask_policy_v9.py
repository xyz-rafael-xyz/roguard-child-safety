import pytest


def test_semantic_selection_keeps_perfect_separation_and_prefers_simple_rule():
    np = pytest.importorskip('numpy')
    from roguard.evidence import ONTOLOGY
    from training.native_fields_v9 import FIELDS, fit_couplings, policy_mass
    from training.select_multitask_fields_v9 import choose
    rows = []
    truth = []
    for source in ('formal', 'v7_training_frame_projection'):
        for concern in ('0', '1'):
            rows.append(dict(source=source, attributes=dict(subject='SELF', time='CURRENT',
                assertion='AFFIRMED', source='DIRECT', relation='NEW', concern=concern)))
            truth.append(int(concern))
    probabilities = [np.array([[float(label == row['attributes'][field])
        for label in ONTOLOGY[field]] for row in rows]) for field in FIELDS]
    best, trials = choose(rows, np.array(truth), probabilities, fit_couplings(rows, [1.] * 4))
    assert len(trials) == 2
    assert best['mode'] == 'independent'
    assert all(value['passes'] for value in best['metrics'].values())
    assert (policy_mass(probabilities) >= best['threshold']).astype(int).tolist() == truth


def test_semantic_selection_rejects_invalid_probabilities():
    np = pytest.importorskip('numpy')
    from training.select_multitask_fields_v9 import choose
    with pytest.raises(ValueError, match='Six field'):
        choose([], np.array([]), [], {})
