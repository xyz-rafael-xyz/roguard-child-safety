import pytest
np=pytest.importorskip('numpy')
from training.native_fields_v9 import policy_mass,SIZES,WORLD_INDICES,WORLD_ACTIONS


def test_policy_mass_matches_all_432_deterministic_worlds():
    values=[np.eye(size)[[w[i] for w in WORLD_INDICES]] for i,size in enumerate(SIZES)]
    np.testing.assert_allclose(policy_mass(values),WORLD_ACTIONS,atol=1e-12)


def test_uniform_world_distribution_and_invalid_inputs():
    values=[np.full((2,size),1/size) for size in SIZES]
    np.testing.assert_allclose(policy_mass(values),sum(WORLD_ACTIONS)/len(WORLD_ACTIONS),atol=1e-12)
    values[0][0,0]=np.nan
    with pytest.raises(ValueError):policy_mass(values)
