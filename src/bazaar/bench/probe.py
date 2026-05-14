"""Linear-probe evaluators.

`run_probe` — regression: StandardScaler + LinearRegression with 1st/99th
percentile target clipping; returns R². Numbers stay directly comparable to
the upstream `pu` regression script.

`run_classification_probe` — classification: StandardScaler + multinomial
LogisticRegression; returns (accuracy, macro-F1). Used by the gz10 bench
for the 10-class morphology label.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, r2_score
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


def run_classification_probe(
    embeddings: np.ndarray,
    y: np.ndarray,
    test_size: int,
    random_state: int,
    max_iter: int = 2000,
) -> tuple[float, float]:
    """Train a multinomial logistic probe; return (accuracy, macro-F1).

    Mirrors `run_probe`'s API (test_size, random_state, valid-mask handling)
    but for integer class labels. Macro-F1 weights each class equally — gz10
    is mildly imbalanced, so the macro average is the more honest
    "did this embedding learn rare classes?" metric. Accuracy is reported
    alongside for direct comparison with paper numbers.
    """
    valid = np.isfinite(embeddings).all(axis=1) & (y >= 0)
    X_v, y_v = embeddings[valid], y[valid].astype(np.int64)
    # Stratify when every class has at least 2 members; otherwise fall back to
    # a plain shuffled split. Stratification fails on the smoke-test scale
    # (rare classes can land 0 or 1 samples after subsampling) but never on
    # the full ~17k gz10 train split where the rarest class still has 100s.
    _, counts = np.unique(y_v, return_counts=True)
    stratify_arg = y_v if counts.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_v, y_v, test_size=test_size, random_state=random_state,
        stratify=stratify_arg,
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    probe = LogisticRegression(
        max_iter=max_iter, solver="lbfgs",
    ).fit(X_tr, y_tr)
    preds = probe.predict(X_te)
    acc = float((preds == y_te).mean())
    f1 = float(f1_score(y_te, preds, average="macro"))
    return acc, f1
