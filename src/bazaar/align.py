"""Alignment primitives for the basket-mean pipeline.

Operations:

- `orthogonal_procrustes(A, B)`: closed-form rotation R minimising ||AR − B||_F.
- `generalized_procrustes(mats)`: iterative GPA — rotate each matrix onto a
  consensus mean until the mean stabilises.
- `mcca_fit(Zs, D)`: Carroll/Kettenring MAX-VAR Generalized CCA — returns the
  projector V (saved in a BazaarFit) and the fit-time shared latent S = C @ V.
- `mcca_transform(Zs, V)`: apply a saved V to new per-model features.
- `mcca_basket(Zs, D)`: back-compat wrapper that returns just the fit-time S.

All operate on already-whitened per-model PCA features (per-feature
z-scored, samples row-aligned). None of them touches the downstream
labels; alignment is unsupervised.
"""
from __future__ import annotations

import numpy as np


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return R such that A @ R ≈ B in Frobenius norm.

    Closed-form: SVD of A^T B = U Σ V^T  ⇒  R = U V^T. R is (D, D)
    orthogonal. Computed in float64 for numerical stability; inputs are
    cast back to their original dtype.
    """
    M = A.T.astype(np.float64) @ B.astype(np.float64)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return (U @ Vt).astype(A.dtype)


def generalized_procrustes(
    mats: list[np.ndarray],
    *, max_iter: int = 50, tol: float = 1e-6, verbose: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Align a list of (N, D) matrices to their consensus mean.

    No scale step: callers should pre-normalise (we use PCA + per-feature
    z-score, so each input has comparable Frobenius norm). No translation
    step: PCA outputs are already mean-centred per feature.

    Returns
    -------
    aligned : list of rotated (N, D) matrices, dtype matches input
    mean    : (N, D) consensus mean
    info    : dict(iterations, loss, loss_per_iter)
    """
    if not mats:
        raise ValueError("generalized_procrustes requires at least one matrix")
    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(f"all matrices must share shape, got {shapes}")

    aligned = [m.copy() for m in mats]
    ref = aligned[0].copy()
    loss_per_iter: list[float] = []

    for it in range(max_iter):
        for i, A in enumerate(aligned):
            R = orthogonal_procrustes(A, ref)
            aligned[i] = A @ R
        new_ref = np.mean(aligned, axis=0)
        loss = float(np.mean([
            np.linalg.norm(a - new_ref) ** 2 for a in aligned
        ]))
        loss_per_iter.append(loss)
        if verbose:
            print(f"    GPA iter {it+1:2d}  mean-Frob² = {loss:.4e}")
        if it > 0 and abs(loss_per_iter[-2] - loss) <= tol * max(loss_per_iter[-2], 1.0):
            ref = new_ref
            break
        ref = new_ref

    return aligned, np.mean(aligned, axis=0), {
        "iterations": len(loss_per_iter),
        "loss": loss_per_iter[-1],
        "loss_per_iter": loss_per_iter,
    }


def mcca_fit(
    Zs: list[np.ndarray], D: int, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Carroll/Kettenring MAX-VAR Generalized CCA.

    Stack horizontally:  C = [Z_1 | Z_2 | ... | Z_M] ∈ R^{N × MD}.
    Randomized SVD:      C ≈ U Σ Vh, retaining n_components = D.

    The MAX-VAR GCCA shared latent on the fit data is S = U * sv = C @ V
    (with V = Vh.T). We return V so that new data with the same per-model
    PCA + z-score preprocessing can be projected via `mcca_transform`.

    Returns
    -------
    V : (MD, D) float32 — projector. Save this in a BazaarFit.
    S : (N, D)  float32 — fit-time shared latent, identical to the legacy
        `mcca_basket(...)` output.
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


def mcca_basket(Zs: list[np.ndarray], D: int, seed: int = 0) -> np.ndarray:
    """Back-compat: return the fit-time shared latent S only."""
    _, S = mcca_fit(Zs, D=D, seed=seed)
    return S
