"""Alignment primitive for the basket-mean pipeline.

- `mcca_fit(Zs, D)`: Carroll/Kettenring MAX-VAR Generalized CCA — returns the
  projector V (saved in a GestaltFit) and the fit-time shared latent S = C @ V.
- `mcca_transform(Zs, V)`: apply a saved V to new per-model features.

Operates on already-whitened per-model PCA features (per-feature
z-scored, samples row-aligned). Unsupervised — never sees labels.
"""

from __future__ import annotations

import numpy as np


def mcca_fit(
    Zs: list[np.ndarray],
    D: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Carroll/Kettenring MAX-VAR Generalized CCA.

    Stack horizontally:  C = [Z_1 | Z_2 | ... | Z_M] ∈ R^{N × MD}.
    Randomized SVD:      C ≈ U Σ Vh, retaining n_components = D.

    The MAX-VAR GCCA shared latent on the fit data is S = U * sv = C @ V
    (with V = Vh.T). We return V so that new data with the same per-model
    PCA + z-score preprocessing can be projected via `mcca_transform`.

    Returns
    -------
    V : (MD, D) float32 — projector. Save this in a GestaltFit.
    S : (N, D)  float32 — fit-time shared latent.
    """
    from sklearn.utils.extmath import randomized_svd

    C = np.concatenate(Zs, axis=1).astype(np.float32)
    U, sv, Vh = randomized_svd(C, n_components=D, random_state=seed)
    V = Vh.T.astype(np.float32)
    S = (U * sv).astype(np.float32)
    return V, S


def mcca_transform(Zs: list[np.ndarray], V: np.ndarray) -> np.ndarray:
    """Apply a saved MCCA projector V to new per-model whitened features.

    `Zs` must use the same basket order as `mcca_fit`. On fit data this is
    bit-for-bit identical to the fit-time S (because C_fit @ V = U Σ Vh V =
    U Σ when V comes from Vh).
    """
    if not Zs:
        raise ValueError("mcca_transform requires at least one view")
    C = np.concatenate(Zs, axis=1).astype(np.float32)
    if C.shape[1] != V.shape[0]:
        raise ValueError(
            f"MCCA projector expects {V.shape[0]} concatenated features; "
            f"got {C.shape[1]} (check basket order/D)."
        )
    return (C @ V).astype(np.float32)
