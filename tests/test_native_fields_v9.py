import pytest
np=pytest.importorskip('numpy')
from training.native_fields_v9 import policy_mass,FIELDS,SIZES,WORLD_INDICES,WORLD_ACTIONS
from roguard.native_fields import factorized_review_mass


def test_policy_mass_matches_all_432_deterministic_worlds():
    values=[np.eye(size)[[w[i] for w in WORLD_INDICES]] for i,size in enumerate(SIZES)]
    np.testing.assert_allclose(policy_mass(values),WORLD_ACTIONS,atol=1e-12)


def test_uniform_world_distribution_and_invalid_inputs():
    values=[np.full((2,size),1/size) for size in SIZES]
    np.testing.assert_allclose(policy_mass(values),sum(WORLD_ACTIONS)/len(WORLD_ACTIONS),atol=1e-12)
    values[0][0,0]=np.nan
    with pytest.raises(ValueError):policy_mass(values)


def test_analytic_batch_and_runtime_match_exhaustive_random_marginals():
    rng=np.random.default_rng(943)
    values=[rng.dirichlet(np.ones(size),size=50) for size in SIZES]
    reference=np.zeros(50)
    for world,action in zip(WORLD_INDICES,WORLD_ACTIONS):
        mass=np.ones(50)
        for p,index in zip(values,world):mass*=p[:,index]
        reference+=action*mass
    np.testing.assert_allclose(policy_mass(values),reference,atol=2e-15,rtol=0)
    actual=[factorized_review_mass({f:p[i].tolist() for f,p in zip(FIELDS,values)}) for i in range(50)]
    np.testing.assert_allclose(actual,reference,atol=2e-15,rtol=0)
