from typing import Sequence

import numpy as np


def check_future_leakage(
    X: np.ndarray, U: np.ndarray, max_lag: int = 50, corr_threshold: float = 0.9
) -> bool:
    """Heuristic: if U is more correlated with future X than past, flag."""
    X = _validate_2d(X)
    U = _validate_2d(U)
    T = min(X.shape[0], U.shape[0])
    X = X[:T]
    U = U[:T]

    x = X[:, 0]
    for j in range(U.shape[1]):
        u = U[:, j]
        # normalize
        xz = (x - x.mean()) / (x.std() + 1e-12)
        uz = (u - u.mean()) / (u.std() + 1e-12)
        # evaluate correlations across lags in [-max_lag, +max_lag]
        best_corr = 0.0
        best_k = 0
        for k in range(-max_lag, max_lag + 1):
            if k >= 0:
                xx = xz[: T - k]
                uu = uz[k:]
            else:
                xx = xz[-k:]
                uu = uz[: T + k]
            if len(uu) < 4:
                continue
            c = float(np.mean(xx * uu))
            if abs(c) > abs(best_corr):
                best_corr = c
                best_k = k
        if best_k < 0 and abs(best_corr) >= corr_threshold:
            return True
    return False


def enforce_past_lags(X: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Build safe lagged matrix with only past lags."""
    X = _validate_2d(X)
    if not lags:
        return X.copy()
    if any(lag <= 0 for lag in lags):
        raise ValueError("lags must be positive (past-only)")
    T, d = X.shape
    max_lag = max(lags)
    cols = [X[max_lag:, :]]
    for lag in sorted(set(lags)):
        cols.append(X[max_lag - lag : T - lag, :])
    return np.hstack(cols)


def _validate_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("Expected 2D array")
    return x
