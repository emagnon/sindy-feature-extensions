import numpy as np
from typing import Iterable, Tuple


def stlsq_threshold(Theta: np.ndarray, y: np.ndarray, thresh: float, max_iter: int = 25) -> np.ndarray:
    """Sparse Thresholded Least Squares (STLSQ) with fixed threshold."""
    Theta = _validate_2d(Theta)
    y = _validate_2d(y)
    # ordinary least squares
    Xi, *_ = np.linalg.lstsq(Theta, y, rcond=None)
    for _ in range(max_iter):
        small = np.abs(Xi) < thresh
        Xi[small] = 0.0
        for k in range(y.shape[1]):
            big = ~small[:, k]
            if np.any(big):
                Xi[big, k], *_ = np.linalg.lstsq(Theta[:, big], y[:, k], rcond=None)
    return Xi


def cv_stlsq(
    Theta: np.ndarray,
    y: np.ndarray,
    thresholds: Iterable[float],
    train_ratio: float = 0.8,
    max_iter: int = 25,
) -> Tuple[np.ndarray, float]:
    """Simple time-preserving split cross-validation for STLSQ threshold selection.

    Returns
    -------
    Xi_best : np.ndarray
        Coefficients at best threshold on validation MSE.
    best_thr : float
        Threshold value minimizing validation MSE.
    """
    Theta = _validate_2d(Theta)
    y = _validate_2d(y)
    n = Theta.shape[0]
    ntr = int(max(2, min(n - 2, np.floor(train_ratio * n))))
    Theta_tr, Theta_va = Theta[:ntr, :], Theta[ntr:, :]
    y_tr, y_va = y[:ntr, :], y[ntr:, :]

    best_err = np.inf
    best_thr = None
    best_Xi = None
    for thr in thresholds:
        Xi = stlsq_threshold(Theta_tr, y_tr, thr, max_iter=max_iter)
        y_hat = Theta_va @ Xi
        err = _rmse(y_va, y_hat)
        if err < best_err:
            best_err = err
            best_thr = float(thr)
            best_Xi = Xi
    if best_Xi is None:
        # fall back to least squares
        best_Xi, *_ = np.linalg.lstsq(Theta_tr, y_tr, rcond=None)
        best_thr = 0.0
    return best_Xi, best_thr


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _validate_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("Expected 2D array")
    return x

