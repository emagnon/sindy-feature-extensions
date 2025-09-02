import numpy as np
from scipy.signal import savgol_filter
from typing import Optional


class FiniteDiff:
    """Centered finite difference with first/second order boundary handling."""

    def __init__(self, order: int = 1):
        if order not in (1, 2):
            raise ValueError("Only first and second derivatives are supported")
        self.order = order

    def differentiate(self, x: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        x = _validate_2d(x)
        if t is None:
            dt = 1.0
        else:
            t = np.asarray(t).flatten()
            if len(t) != x.shape[0]:
                raise ValueError("t must match the number of samples in x")
            dt = np.gradient(t)
        if self.order == 1:
            dxdt = np.vstack([np.gradient(x[:, j], dt) for j in range(x.shape[1])]).T
            return dxdt
        else:
            d2xdt2 = np.vstack([np.gradient(np.gradient(x[:, j], dt), dt) for j in range(x.shape[1])]).T
            return d2xdt2


class SavitzkyGolay:
    """Savitzky–Golay differentiation with automatic window selection."""

    def __init__(self, window_length: int = 11, polyorder: int = 3, deriv: int = 1):
        if window_length % 2 == 0:
            window_length += 1
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def differentiate(self, x: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        x = _validate_2d(x)
        if t is None:
            dt = 1.0
        else:
            t = np.asarray(t).flatten()
            if len(t) != x.shape[0]:
                raise ValueError("t must have same length as x")
            # dt can be non-uniform; we approximate with mean step
            dt = float(np.mean(np.diff(t)))
        out = np.zeros_like(x, dtype=float)
        for j in range(x.shape[1]):
            out[:, j] = savgol_filter(x[:, j], self.window_length, self.polyorder, deriv=self.deriv, delta=dt, mode="interp")
        return out


class TVReg:
    """TV-regularized differentiation (optional dependency on tvregdiff).
    If tvregdiff is not available, raises a clear error at runtime.
    """

    def __init__(self, alph: float = 1.0, itern: int = 100):
        self.alph = alph
        self.itern = itern

    def differentiate(self, x: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        try:
            from tvregdiff import TVRegDiff
        except Exception as e:
            raise ImportError("tvregdiff is required for TVReg. Install with pip install tvregdiff") from e

        x = _validate_2d(x)
        if t is None:
            dt = 1.0
        else:
            t = np.asarray(t).flatten()
            if len(t) != x.shape[0]:
                raise ValueError("t must have same length as x")
            dt = float(np.mean(np.diff(t)))
        out = np.zeros_like(x, dtype=float)
        for j in range(x.shape[1]):
            out[:, j] = TVRegDiff(x[:, j], self.alph, itern=self.itern, dx=dt, scale="small", plotflag=False)
        return out


def _validate_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("Expected 2D array")
    return x
