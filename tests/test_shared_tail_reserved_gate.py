from copy import deepcopy

import pytest

from training.evaluate_shared_tail_reserved_v9 import require_installed_gate
from training.evaluate_canonical_tail_reserved_v9 import require_installed_gate as require_canonical_installed_gate


def good_report():
    groups = {name: dict(positive=150, negative=150, missed=1, false_reviews=7) for name in ('formal', 'conversational', 'exposed')}
    language = dict(replay=dict(decision_changes=0, score_max_error=1e-6), metrics=groups)
    return dict(complete=True, installed_execution_verified=True, prior_counts_pass=True,
        fresh_test_inferred=False, wheel_sha256='a'*64, bundle_sha256='b'*64,
        languages=dict(ro=deepcopy(language), uk=deepcopy(language)))


@pytest.mark.parametrize('require_installed_gate', [require_installed_gate, require_canonical_installed_gate])
def test_reserved_gate_requires_both_languages_every_group_and_exact_installed_identity(require_installed_gate):
    report = good_report()
    require_installed_gate(report, 'a'*64, 'b'*64)
    for key in ('complete', 'installed_execution_verified', 'prior_counts_pass'):
        altered = dict(report, **{key: False})
        with pytest.raises(ValueError):
            require_installed_gate(altered, 'a'*64, 'b'*64)
    for change in ('missing-language', 'missing-group', 'miss', 'false', 'parity', 'nonfinite', 'opened'):
        altered = deepcopy(report)
        if change == 'missing-language':
            del altered['languages']['uk']
        elif change == 'missing-group':
            del altered['languages']['ro']['metrics']['conversational']
        elif change == 'miss':
            altered['languages']['uk']['metrics']['formal']['missed'] = 2
        elif change == 'false':
            altered['languages']['ro']['metrics']['exposed']['false_reviews'] = 8
        elif change == 'parity':
            altered['languages']['uk']['replay']['decision_changes'] = 1
        elif change == 'nonfinite':
            altered['languages']['ro']['replay']['score_max_error'] = float('nan')
        else:
            altered['fresh_test_inferred'] = True
        with pytest.raises(ValueError):
            require_installed_gate(altered, 'a'*64, 'b'*64)
    with pytest.raises(ValueError):
        require_installed_gate(report, 'c'*64, 'b'*64)
