from typing import List, Optional, Sequence

import numpy as np
from pysindy.feature_library.base import BaseFeatureLibrary, x_sequence_or_item
from pysindy.utils import AxesArray


def _check_2d(arr: np.ndarray, name: str = "x") -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for {name}, got shape {arr.shape}")
    return arr


class ChebyshevLibrary(BaseFeatureLibrary):
    """Per-variable Chebyshev polynomial features T_k(x_j)."""

    def __init__(self, degree: int = 4, include_unity: bool = False):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.include_unity = include_unity

    @x_sequence_or_item
    def fit(self, x_full, y=None):
        x0 = x_full[0]
        n_features = x0.shape[x0.ax_coord]
        self.n_features_in_ = n_features
        per_var = (self.degree + 1) if self.include_unity else self.degree
        self.n_output_features_ = n_features * per_var
        return self

    @x_sequence_or_item
    def transform(self, x_full):
        if not hasattr(self, "n_features_in_"):
            raise AttributeError("Call fit before transform")

        xp_full = []
        for x in x_full:
            axes = x.axes
            ax_feat = x.ax_coord
            arr = np.asarray(x)
            _check_2d(arr)

            X = arr if ax_feat == 1 else arr.T
            n_samples, d = X.shape

            cols = []
            for j in range(d):
                xi = X[:, j]
                if self.include_unity:
                    cols.append(np.ones((n_samples, 1)))

                Tkm2 = np.ones_like(xi)
                Tkm1 = xi
                if not self.include_unity:
                    cols.append(Tkm1.reshape(-1, 1))
                for k in range(2, self.degree + 1):
                    Tk = 2.0 * xi * Tkm1 - Tkm2
                    cols.append(Tk.reshape(-1, 1))
                    Tkm2, Tkm1 = Tkm1, Tk

            if cols:
                out_samples_major = np.hstack(cols)
            else:
                out_samples_major = np.empty((X.shape[0], 0))

            out = out_samples_major if ax_feat == 1 else out_samples_major.T
            xp_full.append(AxesArray(out, axes))
        return xp_full

    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        if not hasattr(self, "n_features_in_"):
            raise AttributeError("Call fit before get_feature_names")

        d = self.n_features_in_
        if input_features is None:
            input_features = [f"x{j}" for j in range(d)]

        names = []
        for j in range(d):
            var = input_features[j]
            if self.include_unity:
                names.append(f"T0({var})")
            for k in range(1, self.degree + 1):
                names.append(f"T{k}({var})")
        return names


class TrigLibrary(BaseFeatureLibrary):
    """Per-variable trigonometric features: sin(k x_j), cos(k x_j)."""

    def __init__(self, max_frequency: int = 3, include_base: bool = True):
        if max_frequency < 1:
            raise ValueError("max_frequency must be >= 1")
        self.max_frequency = max_frequency
        self.include_base = include_base

    @x_sequence_or_item
    def fit(self, x_full, y=None):
        x0 = x_full[0]
        n_features = x0.shape[x0.ax_coord]
        self.n_features_in_ = n_features
        if self.include_base:
            per_var = 2 * self.max_frequency
        else:
            per_var = 2 * max(self.max_frequency - 1, 0)
        self.n_output_features_ = n_features * per_var
        return self

    @x_sequence_or_item
    def transform(self, x_full):
        if not hasattr(self, "n_features_in_"):
            raise AttributeError("Call fit before transform")

        xp_full = []
        for x in x_full:
            axes = x.axes
            ax_feat = x.ax_coord
            arr = np.asarray(x)
            _check_2d(arr)

            X = arr if ax_feat == 1 else arr.T
            n_samples, d = X.shape

            cols = []
            for j in range(d):
                xi = X[:, j]
                start_k = 1 if self.include_base else 2
                for k in range(start_k, self.max_frequency + 1):
                    cols.append(np.sin(k * xi).reshape(-1, 1))
                    cols.append(np.cos(k * xi).reshape(-1, 1))

            out_samples_major = np.hstack(cols) if cols else np.empty((X.shape[0], 0))
            out = out_samples_major if ax_feat == 1 else out_samples_major.T
            xp_full.append(AxesArray(out, axes))
        return xp_full

    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        if not hasattr(self, "n_features_in_"):
            raise AttributeError("Call fit before get_feature_names")

        d = self.n_features_in_
        if input_features is None:
            input_features = [f"x{j}" for j in range(d)]

        names = []
        start_k = 1 if self.include_base else 2
        for j in range(d):
            var = input_features[j]
            for k in range(start_k, self.max_frequency + 1):
                if k == 1:
                    names += [f"sin({var})", f"cos({var})"]
                else:
                    names += [f"sin({k}{var})", f"cos({k}{var})"]
        return names


class TimeDelayLibrary(BaseFeatureLibrary):
    """Builds lagged copies of the inputs: x_j(t - lag) for lag in lags."""

    def __init__(
        self,
        lags: Sequence[int],
        include_current: bool = True,
        drop_na: bool = False,
    ):
        if not lags:
            raise ValueError("lags must be a non-empty sequence")
        lags_int = [int(lag) for lag in lags]
        if any(lag <= 0 for lag in lags_int):
            raise ValueError("lags must be positive (past-only)")
        self.lags = sorted(set(lags_int))
        self.include_current = include_current
        self.drop_na = drop_na

    @x_sequence_or_item
    def fit(self, x_full, y=None):
        x0 = x_full[0]
        n_features = x0.shape[x0.ax_coord]
        self.n_features_in_ = n_features
        per_var = (1 if self.include_current else 0) + len(self.lags)
        self.n_output_features_ = n_features * per_var
        return self

    @x_sequence_or_item
    def transform(self, x_full):
        if not hasattr(self, "n_features_in_"):
            raise AttributeError("Call fit before transform")

        xp_full = []
        max_lag = max(self.lags) if self.lags else 0

        for x in x_full:
            axes = x.axes
            ax_feat = x.ax_coord
            arr = np.asarray(x)
            _check_2d(arr)

            X = arr if ax_feat == 1 else arr.T
            n_samples, d = X.shape

            blocks = []
            if self.include_current:
                blocks.append(X)

            for lag in self.lags:
                pad = np.full((lag, d), np.nan)
                lagged = np.vstack([pad, X[:-lag, :]])
                blocks.append(lagged)

            out_samples_major = np.hstack(blocks) if blocks else np.empty((n_samples, 0))

            if self.drop_na and max_lag > 0:
                out_samples_major = out_samples_major[max_lag:, :]

            out = out_samples_major if ax_feat == 1 else out_samples_major.T
            xp_full.append(AxesArray(out, axes))

        return xp_full

    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        if not hasattr(self, "n_features_in_"):
            raise AttributeError("Call fit before get_feature_names")

        d = self.n_features_in_
        if input_features is None:
            input_features = [f"x{j}" for j in range(d)]

        names = []
        if self.include_current:
            for j in range(d):
                names.append(f"{input_features[j]}(t)")
        for lag in self.lags:
            for j in range(d):
                names.append(f"{input_features[j]}(t-{lag})")
        return names
