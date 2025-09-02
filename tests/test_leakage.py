import numpy as np
from sindy_feat_ext.leakage import check_future_leakage, enforce_past_lags


def test_enforce_past_lags():
    X = np.arange(10).reshape(-1, 1)
    out = enforce_past_lags(X, [1, 3])
    assert out.shape[0] == 10 - 3
    assert out.shape[1] == 1 * (1 + 2)


def test_check_future_leakage_flags():
    # x is random walk; u is x shifted negatively (future)
    rng = np.random.default_rng(0)
    e = rng.standard_normal(300)
    x = np.cumsum(e)
    u_future = np.roll(x, -5)  # uses future of x at time t
    leak = check_future_leakage(x[:, None], u_future[:, None], max_lag=10, corr_threshold=0.5)
    assert leak is True

