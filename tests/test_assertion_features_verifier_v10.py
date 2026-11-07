"""Independent feature audit rejects metadata, replay and metric corruption."""
from copy import deepcopy

import pytest

from roguard.assurance import digest
from eval.verify_assertion_features_v10 import check_arrays


def fixture():
    np = pytest.importorskip('numpy')
    original = [{'case_id': 'parent', 'language': 'ro', 'partition': 'train', 'gold': 1,
                 'fields': [0, 0, 2, 0, 0, 1], 'token_sha256': 'a'*64}]
    old_x = np.ones((1, 2560), np.float32); old_z = np.zeros((1, 19), np.float32); old_z[0, -1] = 1.
    parent = {k: original[0][k] for k in ('case_id', 'language', 'gold', 'fields', 'token_sha256')}; parent['feature_index'] = 0
    runtime = [{'case_id': 'new', 'language': 'ro', 'partition': 'train', 'gold': 1,
                'fields': original[0]['fields'][:], 'parent': parent, 'tokens': [1, 2], 'token_sha256': digest([1, 2]),
                'pair_id': 'pair', 'factors': {'operator_index': 0, 'quote_placement': 'before', 'operator_negation_marker': False}}]
    rows = [{'case_id': 'new', 'language': 'ro', 'partition': 'train', 'group': 'assertion_variation_77',
             'gold': 1, 'fields': original[0]['fields'][:], 'token_sha256': digest([1, 2]), 'pair_id': 'pair',
             'parent_feature_index': 0, 'parent_case_id': 'parent', 'parent_token_sha256': 'a'*64,
             'factors': runtime[0]['factors']}]
    parents = [{'feature_index': 0, **original[0]}]
    arrays = {'treatment-features.npy': old_x.copy(), 'treatment-logits.npy': old_z.copy(),
              'control-parent-features.npy': old_x.copy(), 'control-parent-logits.npy': old_z.copy()}
    metrics = {'maximum_parent_feature_error': 0., 'maximum_parent_logit_error': 0.,
               'parent_binary_decision_disagreements': 0, 'treatment_rows': 1, 'control_parent_rows': 1}
    return deepcopy((original, old_x, old_z, runtime, rows, parents, arrays, metrics, {'mode': 'binary', 'threshold': .5}, .001, .001))


def test_exact_saved_anchors_and_counts_reproduce():
    result = check_arrays(*fixture())
    assert result['maximum_parent_feature_error'] == 0 and result['control_parents'] == 1


@pytest.mark.parametrize('fault', ['index', 'dev_parent', 'metadata', 'tokens', 'parent_order', 'shape', 'dtype',
                                 'nonfinite', 'feature_error', 'logit_error', 'metrics', 'decision'])
def test_corrupt_feature_evidence_fails_closed(fault):
    np = pytest.importorskip('numpy')
    values = list(fixture()); original, x, z, runtime, rows, parents, arrays, metrics, spec, _, _ = values
    if fault == 'index': runtime[0]['parent']['feature_index'] = 3
    elif fault == 'dev_parent': original[0]['partition'] = 'dev'
    elif fault == 'metadata': rows[0]['pair_id'] = 'bad'
    elif fault == 'tokens': runtime[0]['tokens'][0] = 9
    elif fault == 'parent_order': parents[0]['feature_index'] = 3
    elif fault == 'shape': arrays['treatment-features.npy'] = arrays['treatment-features.npy'][:, :2]
    elif fault == 'dtype': arrays['treatment-features.npy'] = arrays['treatment-features.npy'].astype(np.float64)
    elif fault == 'nonfinite': arrays['treatment-features.npy'][0, 0] = np.nan
    elif fault == 'feature_error': arrays['control-parent-features.npy'][0, 0] += .01
    elif fault == 'logit_error': arrays['control-parent-logits.npy'][0, 0] += .01
    elif fault == 'metrics': metrics['maximum_parent_feature_error'] = .0001
    else:
        spec['threshold'] = .5; z[0, -1] = 0.; arrays['control-parent-logits.npy'][0, -1] = -.0001
    with pytest.raises(ValueError): check_arrays(*values)
