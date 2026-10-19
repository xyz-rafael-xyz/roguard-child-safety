import pytest


def test_portable_kernel_matches_fitted_rbf_on_nonlinear_boundary():
    np = pytest.importorskip('numpy')
    svm = pytest.importorskip('sklearn.svm')
    from training.gemma4_readout_v9 import normalize
    from training.gemma4_kernel_v9 import margins, scores
    rng = np.random.default_rng(142)
    train = rng.normal(size=(160,8))
    gold = (train[:,0]*train[:,1] > 0).astype(int)
    test = rng.normal(size=(35,8))
    fitted = svm.SVC(C=10., gamma=.1, kernel='rbf').fit(normalize(train),gold)
    head = dict(dual_coefficients=fitted.dual_coef_[0].tolist(),
        intercept=float(fitted.intercept_[0]),gamma=float(fitted._gamma))
    actual = margins(test,head,fitted.support_vectors_)
    assert np.max(np.abs(actual-fitted.decision_function(normalize(test)))) < 1e-10
    assert np.array_equal(scores(test,head,fitted.support_vectors_) >= .5,fitted.predict(normalize(test)) == 1)
    with pytest.raises(ValueError):
        margins(test,dict(head,gamma=-1),fitted.support_vectors_)
    with pytest.raises(ValueError):
        margins(test,head,fitted.support_vectors_[:,:7])
    with pytest.raises(ValueError):
        margins(test,dict(head,dual_coefficients=[float('nan')]*len(fitted.support_)),fitted.support_vectors_)
