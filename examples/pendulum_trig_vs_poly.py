import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps  # pip install pysindy

from sindy_feat_ext.derivatives import SavitzkyGolay
from sindy_feat_ext.libraries import TrigLibrary
from sindy_feat_ext.sparsify import cv_stlsq

# -----------------------------
# 1) Simulate a damped pendulum
# -----------------------------
# True system:
#   theta_dot = omega
#   omega_dot = - g/L * sin(theta) - c * omega


def pendulum_rhs(x, g_over_L=9.81, c=0.1):
    theta, omega = x
    return np.array([omega, -g_over_L * np.sin(theta) - c * omega], dtype=float)


def rk4_step(x, dt, rhs):
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_pendulum(T=40.0, dt=0.01, x0=(1.2, 0.0)):
    t = np.arange(0.0, T + dt, dt)
    X = np.zeros((t.size, 2), dtype=float)
    X[0] = np.array(x0, dtype=float)
    for k in range(1, t.size):
        X[k] = rk4_step(X[k - 1], dt, pendulum_rhs)
    return t, X


# -----------------------------
# 2) Prepare data (train/test)
# -----------------------------
rng = np.random.default_rng(42)
dt = 0.01
t, X_true = simulate_pendulum(T=40.0, dt=dt, x0=(1.2, 0.0))

# Add small measurement noise
noise_level = 0.01  # 1% noise
X_noisy = X_true.copy()
X_noisy[:, 0] += noise_level * np.std(X_true[:, 0]) * rng.standard_normal(X_true.shape[0])
X_noisy[:, 1] += 0.3 * noise_level * np.std(X_true[:, 1]) * rng.standard_normal(X_true.shape[0])

# Train/test split along time
n = X_noisy.shape[0]
n_train = int(0.6 * n)
Xtr, Xte = X_noisy[:n_train], X_noisy[n_train:]
t_tr, t_te = t[:n_train], t[n_train:]
Xtrue_te = X_true[n_train:]  # for test evaluation

# Derivatives using Savitzky-Golay (robust to noise)
sg = SavitzkyGolay(window_length=51, polyorder=5, deriv=1)
Xdot = sg.differentiate(X_noisy, t)
Xdtr, Xdte = Xdot[:n_train], Xdot[n_train:]


# --------------------------------------------
# 3) Utilities: scaling, fitting, pretty print
# --------------------------------------------
def scale_columns(Theta):
    arr = np.asarray(Theta)
    s = arr.std(axis=0, ddof=1)
    s[s == 0] = 1.0
    return arr / s, s


def fit_sindy_with_cv(Theta, Y, thresholds):
    # Scale columns, fit with CV STLSQ, then unscale coefficients
    Theta_s, s = scale_columns(Theta)
    Xi_s, thr = cv_stlsq(Theta_s, Y, thresholds=thresholds)
    # Xi_s shape: (n_features, n_targets); s shape: (n_features,)
    # Divide each row of Xi_s by the corresponding s
    Xi = np.asarray(Xi_s) / s[:, None]
    return Xi, thr


def nonzeros(vec_or_mat, tol=1e-8):
    return int(np.sum(np.abs(vec_or_mat) > tol))


def pretty_equation(names, coeffs, eq_name, tol=1e-8):
    # coeffs: shape (n_features,) or (n_features, 1)
    coeffs = np.ravel(coeffs)
    terms = []
    for c, n in zip(coeffs, names):
        if abs(c) > tol:
            terms.append(f"{c:+.4f}*{n}")
    rhs = " ".join(terms) if terms else "0"
    return f"{eq_name} = {rhs}"


# ------------------------------------------------------------
# 4) Baseline: PolynomialLibrary (no domain knowledge)
# ------------------------------------------------------------
poly = ps.PolynomialLibrary(degree=5, include_interaction=True, include_bias=False)
Theta_poly_tr = poly.fit_transform(Xtr)
names_poly = poly.get_feature_names()

Xi_poly, thr_poly = fit_sindy_with_cv(Theta_poly_tr, Xdtr, thresholds=np.logspace(-6, -1, 12))

print("[Baseline: PolynomialLibrary]")
print("  Best threshold:", thr_poly)
print("  Nonzeros (theta_dot):", nonzeros(Xi_poly[:, 0]))
print("  Nonzeros (omega_dot):", nonzeros(Xi_poly[:, 1]))
print(" ", pretty_equation(names_poly, Xi_poly[:, 0], "theta_dot"))
print(" ", pretty_equation(names_poly, Xi_poly[:, 1], "omega_dot"))


# ------------------------------------------------------------
# 5) Extension: TrigLibrary on theta + Identity on omega
# ------------------------------------------------------------
trig = TrigLibrary(max_frequency=1, include_base=True)
ident = ps.feature_library.IdentityLibrary()

# Apply trig only to column 0 (theta), identity only to column 1 (omega)
lib_ext = ps.feature_library.GeneralizedLibrary(
    libraries=[trig, ident],
    inputs_per_library=[[0], [1]],
)

Theta_ext_tr = lib_ext.fit_transform(Xtr)
names_ext = lib_ext.get_feature_names()

Xi_ext, thr_ext = fit_sindy_with_cv(Theta_ext_tr, Xdtr, thresholds=np.logspace(-6, -1, 12))

print("\n[Extension: Trig on theta + Identity on omega]")
print("  Best threshold:", thr_ext)
print("  Nonzeros (theta_dot):", nonzeros(Xi_ext[:, 0]))
print("  Nonzeros (omega_dot):", nonzeros(Xi_ext[:, 1]))
print(" ", pretty_equation(names_ext, Xi_ext[:, 0], "theta_dot"))
print(" ", pretty_equation(names_ext, Xi_ext[:, 1], "omega_dot"))


# ------------------------------------------------------------
# 6) Evaluate forecasting on the held-out test segment
# ------------------------------------------------------------
def make_rhs_from_library(lib, Xi):
    # Returns a callable f(x) that computes xdot = Theta(x) @ Xi
    def f(x):
        X1 = x.reshape(1, -1)
        Theta = lib.transform(X1)  # (1, n_features)
        return (np.asarray(Theta) @ Xi).ravel()

    return f


# Build RHS functions
rhs_poly = make_rhs_from_library(poly, Xi_poly)
rhs_ext = make_rhs_from_library(lib_ext, Xi_ext)


def simulate_model(x0, rhs, t_grid, dt):
    X_sim = np.zeros((t_grid.size, 2))
    X_sim[0] = x0
    for k in range(1, t_grid.size):
        X_sim[k] = rk4_step(X_sim[k - 1], dt, rhs)
    return X_sim


x0_test = X_noisy[n_train]
X_poly_te = simulate_model(x0_test, rhs_poly, t_te, dt)
X_ext_te = simulate_model(x0_test, rhs_ext, t_te, dt)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


rmse_poly_theta = rmse(X_poly_te[:, 0], Xtrue_te[:, 0])
rmse_ext_theta = rmse(X_ext_te[:, 0], Xtrue_te[:, 0])
rmse_poly_omega = rmse(X_poly_te[:, 1], Xtrue_te[:, 1])
rmse_ext_omega = rmse(X_ext_te[:, 1], Xtrue_te[:, 1])

print("\n[Test RMSE vs ground truth (lower is better)]")
print(f"  PolynomialLibrary   -> theta: {rmse_poly_theta:.4f}, omega: {rmse_poly_omega:.4f}")
print(f"  Trig+Identity (ext) -> theta: {rmse_ext_theta:.4f},  omega: {rmse_ext_omega:.4f}")


# ------------------------------------------------------------
# 7) Plot: truth vs identified models on test segment
# ------------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axs[0].plot(t_te, Xtrue_te[:, 0], "k", lw=2, label="theta (true)")
axs[0].plot(t_te, X_poly_te[:, 0], "r--", label="theta (poly)")
axs[0].plot(t_te, X_ext_te[:, 0], "b-.", label="theta (trig+id)")
axs[0].set_ylabel("theta")
axs[0].legend(loc="upper right")

axs[1].plot(t_te, Xtrue_te[:, 1], "k", lw=2, label="omega (true)")
axs[1].plot(t_te, X_poly_te[:, 1], "r--", label="omega (poly)")
axs[1].plot(t_te, X_ext_te[:, 1], "b-.", label="omega (trig+id)")
axs[1].set_ylabel("omega")
axs[1].set_xlabel("time")
axs[1].legend(loc="upper right")

plt.suptitle("Damped Pendulum: SINDy with Custom Features vs Baseline")
plt.tight_layout()
plt.show()
