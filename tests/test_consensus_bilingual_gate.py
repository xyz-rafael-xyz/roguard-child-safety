from copy import deepcopy
import pytest
from training.verify_consensus_bilingual_v9 import validate_candidate


def candidate(language):
    groups = {kind: dict(positive=100, negative=100, missed=1, false_reviews=5, abstentions=0)
              for kind in ('formal', 'conversational')}
    selected = dict(model={'fixed': 'model'}, threshold=.4)
    if language == 'ro':
        selected['development'] = dict(oof=deepcopy(groups), refit=deepcopy(groups))
    else:
        selected.update(development=deepcopy(groups), oof=deepcopy(groups))
    audit = dict(calibration=deepcopy(groups), counts_pass=True, selection_sha256='selected')
    actual = dict(actual_models_loaded=True, cached_component_outputs_used=False,
                  parity_pass=True, bundle_sha256='bundle', rows=200,
                  decision_changes=0, maximum_score_error=0.)
    if language == 'uk':
        actual.update(complete=True, selection_sha256='selected', stress_pass=True,
            splits=dict(screen=dict(rows=200, parity_pass=True, decision_changes=0, maximum_score_error=0.),
                stress=dict(counts=dict(rows=216, positive=44, negative=172, missed=0,
                                       false_reviews=8, abstentions=0))))
    fusion = dict(language=language, selection_sha256='selected', model=selected['model'], threshold=.4)
    if language == 'uk':
        selected.update(threshold=.5,native_threshold=.01,base_threshold=.4,rescue_threshold=.85)
        fusion=dict(format='roguard-scope-consensus-artifact-v1',selection_sha256='selected',native_threshold=.01,base_threshold=.4,rescue_threshold=.85)
    return dict(selection=selected, audit=audit, actual=actual, fusion=fusion,
                selection_sha='selected', bundle_sha='bundle')


@pytest.mark.parametrize('language', ['ro', 'uk'])
def test_passing_candidate_evidence_is_accepted(language):
    validate_candidate(language, **candidate(language))


@pytest.mark.parametrize('language', ['ro', 'uk'])
@pytest.mark.parametrize('failure', ['calibration', 'oof', 'parity', 'cached', 'model', 'threshold', 'bundle'])
def test_reserved_gate_rejects_failed_or_rebound_candidates(language, failure):
    value = candidate(language)
    if failure == 'calibration':
        value['audit']['calibration']['conversational']['missed'] = 2
    elif failure == 'oof':
        groups = value['selection']['development']['oof'] if language == 'ro' else value['selection']['oof']
        groups['formal']['false_reviews'] = 6
    elif failure == 'parity':
        value['actual']['parity_pass'] = False
    elif failure == 'cached':
        value['actual']['cached_component_outputs_used'] = True
    elif failure == 'model':
        value['fusion']['model' if language=='ro' else 'format'] = {'other': 'model'}
    elif failure == 'threshold':
        value['fusion']['threshold' if language=='ro' else 'rescue_threshold'] = .3
    else:
        value['actual']['bundle_sha256'] = 'other'
    with pytest.raises(ValueError):
        validate_candidate(language, **value)


@pytest.mark.parametrize('key,value', [('rows', 215), ('missed', 1), ('false_reviews', 9),
                                     ('positive', True), ('abstentions', -1)])
def test_ukrainian_stress_cannot_be_skipped_or_rounded_into_a_pass(key, value):
    evidence = candidate('uk')
    evidence['actual']['splits']['stress']['counts'][key] = value
    with pytest.raises(ValueError, match='stress'):
        validate_candidate('uk', **evidence)
