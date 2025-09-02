import numpy as np
from typing import List, Optional, Sequence


class _BaseLibrary:
    """Minimal PySINDy-like feature library interface."""

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        x = _validate_2d(x)
        self.n_input_features_ = x.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(x, y).transform(x)

    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        raise NotImplementedError


def _validate_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("Expected 2D array of shape (n_samples, n_features)")
    return x


class ChebyshevLibrary(_BaseLibrary):
    """Per-variable Chebyshev polynomial features T_k(x_i).
    degree=4 yields T1..T4 per feature. Optionally drop the bias term.
    """

    def __init__(self, degree: int = 4, include_unity: bool = False):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.include_unity = include_unity

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = _validate_2d(x)
        feats = []
        names = []
        for j in range(x.shape[1]):
            xi = x[:, j]
            for k in range(1 if not self.include_unity else 0, self.degree + 1):
                if k == 0:
                    val = np.ones_like(xi)
                    name = f"T0(x{j})"
                else:
                    # evaluate per-sample Chebyshev via recurrence
                    # T0=1, T1=t, Tk=2 t T_{k-1} - T_{k-2}
                    Tkm2 = np.ones_like(xi)
                    Tkm1 = xi.copy()
                    if k == 1:
                        Tk = Tkm1
                    else:
                        Tk = None
                        for _ in range(2, k + 1):
                            Tk = 2 * xi * Tkm1 - Tkm2
                            Tkm2, Tkm1 = Tkm1, Tk
                    val = Tk
                    name = f"T{k}(x{j})"
                feats.append(val)
                names.append(name)
        self._feature_names = names
        return np.vstack(feats).T

    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        if input_features is None:
            return self._feature_names
        # remap x{j} to provided names
        out = []
        for n in self._feature_names:
            if n.startswith("T0("):
                out.append(n.replace("x0", input_features[0]))  # rarely used
                continue
            # find index between 'x' and ')'
            start = n.find("x") + 1
            end = n.find(")")
            j = int(n[start:end])
            out.append(n.replace(f"x{j}", input_features[j]))
        return out


class TrigLibrary(_BaseLibrary):
    """Per-variable trigonometric features: sin(k x_i), cos(k x_i)."""

    def __init__(self, max_frequency: int = 3, include_base: bool = True):
        if max_frequency < 1:
            raise ValueError("max_frequency must be >= 1")
        self.max_frequency = max_frequency
        self.include_base = include_base

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = _validate_2d(x)
        feats = []
        names = []
        for j in range(x.shape[1]):
            xi = x[:, j]
            if self.include_base:
                feats.append(np.sin(xi))
                feats.append(np.cos(xi))
                names += [f"sin(x{j})", f"cos(x{j})"]
            for k in range(2, self.max_frequency + 1):
                feats.append(np.sin(k * xi))
                feats.append(np.cos(k * xi))
                names += [f"sin({k}x{j})", f"cos({k}x{j})"]
        self._feature_names = names
        return np.vstack(feats).T if feats else np.empty((x.shape[0], 0))

    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        if input_features is None:
            return self._feature_names
        out = []
        for n in self._feature_names:
            start = n.find("x") + 1
            end = n.find(")", start) if ")" in n else len(n)
            j = int(n[start:end])
            out.append(n.replace(f"x{j}", input_features[j]))
        return out


class TimeDelayLibrary(_BaseLibrary):
    """Builds past-lagged copies of the inputs: x(t - l) for l in lags.

    Parameters
    ----------
    lags : Sequence[int]
        Positive integers interpreted as number of steps back (no future allowed).
    include_current : bool
        If True, also include x(t) alongside lagged features.
    drop_na : bool
        If True, discards the first max(lags) rows so output aligns without NaNs.
    """

    def __init__(self, lags: Sequence[int], include_current: bool = True, drop_na: bool = True):
        if not len(lags):
            raise ValueError("lags must be a non-empty sequence")
        if any(l <= 0 for l in lags):
            raise ValueError("lags must be positive (past-only)")
        self.lags = sorted(set(int(l) for l in lags))
        self.include_current = include_current
        self.drop_na = drop_na

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = _validate_2d(x)
        T, d = x.shape
        max_lag = max(self.lags)
        cols = []
        names = []
        if self.include_current:
            cols.append(x)
            names += [f"x{j}(t)" for j in range(d)]
        for l in self.lags:
            pad = np.full((l, d), np.nan)
            lagged = np.vstack([pad, x[:-l, :]])
            cols.append(lagged)
            names += [f"x{j}(t-{l})" for j in range(d)]
        out = np.hstack(cols)
        if self.drop_na:
            out = out[max_lag:, :]
            self._offset = max_lag
        else:
            self._offset = 0
        self._feature_names = names
        return out

    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        if input_features is None:
            return self._feature_names
        out = []
        for name in self._feature_names:
            # find indices like x{j}
            start = name.find("x") + 1
            end = name.find("(", start)
            j = int(name[start:end])
            out.append(name.replace(f"x{j}", input_features[j]))
        return out
