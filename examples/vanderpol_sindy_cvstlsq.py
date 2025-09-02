import numpy as np
import matplotlib.pyplot as plt

try:
    import pysindy as ps
except ImportError:
    raise SystemExit("Install pysindy to run this example: pip install pysindy")

from sindy_feat_ext.libraries import ChebyshevLibrary, TimeDelayLibrary
from sindy_feat_ext.derivatives import SavitzkyGolay
from sindy_feat_ext.sparsify import cv_stlsq


def van_der_pol(mu, x):
    return np.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


def simulate_vdp(mu=1.2, dt=0.01, T=50.0):
    t = np.arange(0, T + dt, dt)
    x = np.zeros((t.size, 2))
    x[0] = [0.1, 2.0]
    for k in range(1, t.size):
        x[k] = x[k - 1] + dt * van_der_pol(mu, x[k - 1])
    return t, x


if __name__ == "__main__":
    t, x = simulate_vdp()
    der = SavitzkyGolay(window_length=31, polyorder=5).differentiate(x, t)

    # Build feature library
    poly = ps.PolynomialLibrary(degree=3, include_interaction=True)
    cheb = ChebyshevLibrary(degree=4, include_unity=False)
    td = TimeDelayLibrary(lags=[1, 2], include_current=True)

    lib = ps.feature_library.GeneralizedLibrary([poly, cheb, td])
    Theta = lib.fit_transform(x)

    Xi, thr = cv_stlsq(Theta, der[:, [0]], thresholds=np.logspace(-4, -2, 6))
    print("Best threshold:", thr)
    print("Nonzeros:", np.flatnonzero(np.abs(Xi) > 1e-8).size)

    plt.plot(t, x[:, 0], label="x1")
    plt.plot(t, der[:, 0], label="dx1/dt (sg)")
    plt.legend()
    plt.show()
