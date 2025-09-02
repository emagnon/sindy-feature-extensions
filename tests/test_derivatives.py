import numpy as np
from sindy_feat_ext.derivatives import FiniteDiff, SavitzkyGolay


def test_finite_diff_constant():
    t = np.linspace(0, 1, 101)
    x = np.ones((t.size, 2))
    dx = FiniteDiff().differentiate(x, t)
    assert np.allclose(dx, 0.0, atol=1e-8)


def test_savgol_sine():
    t = np.linspace(0, 2*np.pi, 501)
    x = np.sin(t)[:, None]
    dx = SavitzkyGolay(window_length=31, polyorder=5).differentiate(x, t)
    assert dx.shape == x.shape
    # derivative cos(t) approx; correlation positive
    corr = np.corrcoef(dx[:, 0], np.cos(t))[0, 1]
    assert corr > 0.95

