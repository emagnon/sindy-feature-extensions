import numpy as np
from sindy_feat_ext.libraries import ChebyshevLibrary, TrigLibrary, TimeDelayLibrary


def test_chebyshev_shapes():
    x = np.random.randn(100, 2)
    lib = ChebyshevLibrary(degree=3)
    Phi = lib.fit_transform(x)
    assert Phi.shape == (100, 2 * 3)
    names = lib.get_feature_names(["a", "b"])
    assert len(names) == Phi.shape[1]
    assert any("T3(a)" in n or "T3(b)" in n for n in names)


def test_trig_shapes():
    x = np.random.randn(50, 3)
    lib = TrigLibrary(max_frequency=2, include_base=True)
    Phi = lib.fit_transform(x)
    # per feature: sin, cos, sin2, cos2 -> 4 features
    assert Phi.shape == (50, 3 * 4)


def test_time_delay():
    x = np.arange(10).reshape(-1, 1)  # 10x1
    lib = TimeDelayLibrary(lags=[1, 3], include_current=True, drop_na=True)
    Phi = lib.fit_transform(x)
    # T=10, maxlag=3 => 7 rows, columns = current + 2 lags = 3
    assert Phi.shape == (7, 3)

