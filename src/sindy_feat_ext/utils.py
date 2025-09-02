import numpy as np
from typing import Generator, Tuple


def time_delay_embed(x: np.ndarray, lags: int) -> np.ndarray:
    """Create a Hankel-like embedding with lags past steps for each feature.

    Output shape: (n_samples - lags, (lags+1)*n_features)
    """
    x = _validate_2d(x)
    T, d = x.shape
    if lags < 0:
        raise ValueError("lags must be >= 0")
    out = []
    for k in range(lags + 1):
        out.append(x[lags - k : T - k, :])
    return np.hstack(out)


def rolling_window(a: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """Rolling window view along first axis."""
    a = np.asarray(a)
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    n = a.shape[0]
    idxs = [range(i, i + window) for i in range(0, n - window + 1, step)]
    return np.array([a[idx] for idx in idxs])


def rolling_time_split(n_samples: int, initial: int, step: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield expanding-window train/test indices for time series."""
    if initial <= 0 or step <= 0:
        raise ValueError("initial and step must be positive")
    start = initial
    while start + step <= n_samples:
        train_idx = np.arange(start)
        test_idx = np.arange(start, min(n_samples, start + step))
        yield train_idx, test_idx
        start += step


def _validate_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("Expected 2D array")
    return x

