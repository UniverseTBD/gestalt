"""Per-model whitening primitives.

These are the per-(family, size) preprocessing steps applied to raw foundation-
model embeddings before MCCA. Kept tiny and dependency-free so `BazaarFit`
remains self-contained without dragging in the benchmark suite.

- `pca_zscore_fit` / `pca_zscore_transform` — randomized-SVD PCA to D, then
  per-feature z-score. Equal D per model.
- `zscore_fit` / `zscore_transform` — per-feature z-score only. Per-model
  widths stay at native d_m.

Both fit functions return `(Z, artifacts)` where `artifacts` is a small dict
of numpy arrays sufficient to reproduce `Z` from a new embedding matrix via
the matching `_transform`. Float32 throughout.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def pca_zscore_fit(
    E: np.ndarray, D: int, seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit per-model PCA + per-feature z-score on E.

    Returns
    -------
    Z         : (N, D_eff) float32 — whitened features on the fit data.
    artifacts : dict with keys
                  - pca_components : (D_eff, d_in)
                  - pca_mean       : (d_in,)
                  - zscore_mu      : (1, D_eff)
                  - zscore_sd      : (1, D_eff)
                Sufficient to reproduce Z via `pca_zscore_transform`.
    """
    D_eff = min(D, E.shape[1], E.shape[0])
    pca = PCA(n_components=D_eff, svd_solver="randomized", random_state=seed)
    Z = pca.fit_transform(E)
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True) + 1e-8
    artifacts = {
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean":       pca.mean_.astype(np.float32),
        "zscore_mu":      mu.astype(np.float32),
        "zscore_sd":      sd.astype(np.float32),
    }
    return ((Z - mu) / sd).astype(np.float32), artifacts


def pca_zscore_transform(E: np.ndarray, artifacts: dict[str, np.ndarray]) -> np.ndarray:
    """Apply saved PCA + z-score artifacts to (possibly new) embeddings."""
    E64 = E.astype(np.float32, copy=False)
    Z = (E64 - artifacts["pca_mean"]) @ artifacts["pca_components"].T
    return ((Z - artifacts["zscore_mu"]) / artifacts["zscore_sd"]).astype(np.float32)


def pca_and_zscore(E: np.ndarray, D: int, seed: int = 0) -> np.ndarray:
    """Back-compat: PCA on full E (randomized SVD), then per-feature z-score."""
    Z, _ = pca_zscore_fit(E, D=D, seed=seed)
    return Z


def zscore_fit(E: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit per-feature z-score on raw embeddings (no PCA, no dim reduction).

    The PCA-free counterpart of `pca_zscore_fit`. Output keeps `E`'s column
    count, so downstream MCCA receives heterogeneous-width per-model views —
    handled fine by the SVD.
    """
    E32 = E.astype(np.float32, copy=False)
    mu = E32.mean(axis=0, keepdims=True)
    sd = E32.std(axis=0, keepdims=True) + 1e-8
    artifacts = {
        "zscore_mu": mu.astype(np.float32),
        "zscore_sd": sd.astype(np.float32),
    }
    return ((E32 - mu) / sd).astype(np.float32), artifacts


def zscore_transform(E: np.ndarray, artifacts: dict[str, np.ndarray]) -> np.ndarray:
    """Apply saved per-feature z-score stats to (possibly new) embeddings."""
    E32 = E.astype(np.float32, copy=False)
    return ((E32 - artifacts["zscore_mu"]) / artifacts["zscore_sd"]).astype(np.float32)
