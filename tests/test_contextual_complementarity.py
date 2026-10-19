import math

import pytest

from training.diagnose_contextual_complementarity_v9 import blend


def test_logit_pool_matches_odds_and_handles_saturation():
    score = blend(.8, .2, .25, 'logit')
    odds = 4 ** .25 * .25 ** .75
    assert score == pytest.approx(odds / (1 + odds))
    assert blend(0., 1., .5, 'logit') == pytest.approx(.5, abs=1e-5)
    for mode in ('probability', 'logit'):
        for a, b in ((0., 0.), (1., 1.), (.1, .9)):
            value = blend(a, b, .25, mode)
            assert math.isfinite(value) and 0 <= value <= 1
        assert blend(.8, .2, .75, mode) > blend(.8, .2, .25, mode)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1., 2.])
def test_invalid_scores_rejected(value):
    with pytest.raises(ValueError):
        blend(value, .5, .5, 'logit')
