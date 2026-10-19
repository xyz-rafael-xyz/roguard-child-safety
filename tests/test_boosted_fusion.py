import pytest
from training.boosted_fusion_v9 import export_model, probability


def test_portable_trees_match_sklearn_on_unseen_vectors():
    np=pytest.importorskip('numpy')
    ensemble=pytest.importorskip('sklearn.ensemble')
    rng=np.random.default_rng(978)
    x=rng.normal(size=(160,4));y=((x[:,0]>0)^(x[:,1]>0)).astype(int)
    fitted=ensemble.GradientBoostingClassifier(n_estimators=20,max_depth=2,random_state=978).fit(x,y)
    model=export_model(fitted)
    probes=rng.normal(size=(40,4))
    assert max(abs(float(a)-probability(model,b)) for a,b in zip(fitted.predict_proba(probes)[:,1],probes))<1e-12
    assert probability(model,None) is None
    with pytest.raises(ValueError):probability(model,[0.])
    with pytest.raises(ValueError):probability(model,[float('nan'),0.,0.,0.])
