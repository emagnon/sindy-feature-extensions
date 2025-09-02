# SINDy Feature Extensions

This work was realized in 2019 and was designed to extend the original pySINDy repo: https://github.com/luckystarufo/pySINDy version 0.2.0

Small, reusable add-ons for SINDy-style system identification:
- Feature libraries: Chebyshev polynomials, per-variable trigonometric, time-delay embeddings
- Derivative estimators: finite difference, Savitzky–Golay, optional TV-regularized
- Cross-validated sparse regression (CV-STLSQ) for robust threshold selection
- Leakage guards: detect future-looking inputs, safe lag builders
- Utilities: roll-forward splits and embedding helpers

Compatible with PySINDy-style APIs (fit/transform/get_feature_names). You can use these with or without PySINDy.

## Install (dev)
```bash
git clone https://github.com/emagnon/sindy-feature-extensions.git
cd sindy-feature-extensions
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## Quickstart
```python
import numpy as np
from sindy_feat_ext.libraries import ChebyshevLibrary, TimeDelayLibrary, TrigLibrary
from sindy_feat_ext.derivatives import FiniteDiff, SavitzkyGolay
from sindy_feat_ext.sparsify import cv_stlsq
from sindy_feat_ext.leakage import enforce_past_lags
from sindy_feat_ext.utils import time_delay_embed

# data
t = np.linspace(0, 20, 2001)
x = np.vstack([np.sin(t), np.cos(t)]).T  # (T, n_features)

# feature libraries (can be composed manually or via PySINDy GeneralizedLibrary)
cheb = ChebyshevLibrary(degree=4, include_unity=False)
trig = TrigLibrary(max_frequency=3)
td = TimeDelayLibrary(lags=[1,2,5], include_current=True)

Phi = np.hstack([cheb.fit_transform(x), trig.fit_transform(x), td.fit_transform(x)])

# derivatives
dxdt = FiniteDiff().differentiate(x, t)  # or SavitzkyGolay().differentiate(x, t)

# cross-validated sparse regression (threshold search)
coef, best_thr = cv_stlsq(Phi, dxdt[:, [0]], thresholds=np.logspace(-4, -1, 8), train_ratio=0.8)
print(best_thr, coef.shape)
```

## With PySINDy (optional)
```python
import pysindy as ps
from sindy_feat_ext.libraries import ChebyshevLibrary, TimeDelayLibrary

poly = ps.PolynomialLibrary(degree=2)
cheb = ChebyshevLibrary(degree=4, include_unity=False)
td = TimeDelayLibrary(lags=[1,2,5], include_current=True)

lib = ps.feature_library.GeneralizedLibrary([poly, cheb, td])
model = ps.SINDy(feature_library=lib)  # choose your optimizer as usual
```
