import numpy as np
from sindy_feat_ext.sparsify import stlsq_threshold, cv_stlsq


def test_stlsq_identity():
    rng = np.random.default_rng(0)
    Theta = rng.standard_normal((200, 5))
    true_Xi = np.array([[1.0], [0.0], [2.0], [0.0], [0.0]])
    y = Theta @ true_Xi + 1e-3 * rng.standard_normal((200, 1))
    Xi = stlsq_threshold(Theta, y, thresh=0.1)
    nz = np.where(np.abs(Xi[:, 0]) > 1e-2)[0]
    assert set(nz) <= {0, 2}


def test_cv_stlsq_runs():
    rng = np.random.default_rng(1)
    Theta = rng.standard_normal((300, 10))
    Xi_true = np.zeros((10, 1)); Xi_true[1, 0] = 1.5; Xi_true[7, 0] = -0.7
    y = Theta @ Xi_true + 0.01 * rng.standard_normal((300, 1))
    Xi, thr = cv_stlsq(Theta, y, thresholds=np.logspace(-3, -1, 5))
    assert Xi.shape == (10, 1)
    assert thr > 0

