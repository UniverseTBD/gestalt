"""Linear-probe R² evaluator.

The probe is intentionally tiny: StandardScaler + LinearRegression on the
train split, R² on the held-out test split, with 1st/99th-percentile
clipping of the target. The numbers it produces are directly comparable
to those reported in the upstream `pu` regression script.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_probe(
    embeddings: np.ndarray,
    y: np.ndarray,
    test_size: int,
    random_state: int,
) -> float:
    """Train a linear probe on `embeddings` predicting `y`. Return test R²."""
    valid = np.isfinite(y)
    X_v, y_v = embeddings[valid], y[valid]
    lo, hi = np.quantile(y_v, [0.01, 0.99])
    y_clipped = np.clip(y_v, lo, hi)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_v, y_clipped, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    probe = LinearRegression().fit(X_tr, y_tr)
    return float(r2_score(y_te, probe.predict(X_te)))
