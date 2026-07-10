"""Probe-coefficient helper for `bazaar bench probes`.

Mirrors `bazaar.bench.linear_probe.run_probe`'s pipeline (1st/99th percentile clip
on the target, StandardScaler on features, LinearRegression) but returns the
*un-scaled* coefficient vector + intercept + valid count instead of a held-
out R². The un-scaling — dividing by `scaler.scale_` — is what makes
`cos(w_A, w_B)` meaningful across two probes trained in different feature
spaces (or with different per-dim scales), because it puts both probe
vectors back into the raw embedding space rather than the standardized one.

This helper is the only place we need probe weight vectors. The eval-style
held-out R² stays in `run_probe`.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def fit_probe_coeffs(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit a clip+scale+linear probe and return the un-scaled weight vector.

    Returns
    -------
    dict with keys
        w        : (D,) float32 — un-scaled coefficient vector in X's
                   native coordinate system.
        b        : float        — intercept in raw space.
        n_valid  : int          — rows with finite y used.
    """
    valid = np.isfinite(y)
    X_v = X[valid].astype(np.float32, copy=False)
    y_v = y[valid].astype(np.float32, copy=False)
    if X_v.shape[0] < 3:
        raise ValueError(
            f"fit_probe_coeffs: only {X_v.shape[0]} finite rows — "
            f"can't fit a probe."
        )

    lo, hi = np.quantile(y_v, [0.01, 0.99])
    y_clipped = np.clip(y_v, lo, hi)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_v)
    probe = LinearRegression().fit(X_scaled, y_clipped)

    # w_scaled / scaler.scale_ recovers the coefficient on the raw feature.
    w_unscaled = (probe.coef_ / scaler.scale_).astype(np.float32)
    # b_raw = b_scaled - dot(w_scaled / scale_, mean_).
    b_unscaled = float(probe.intercept_ - np.dot(w_unscaled, scaler.mean_))
    return {
        "w": w_unscaled,
        "b": b_unscaled,
        "n_valid": int(X_v.shape[0]),
    }


__all__ = ["fit_probe_coeffs"]
