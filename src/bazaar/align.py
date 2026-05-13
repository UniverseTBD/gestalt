"""Alignment primitives for the basket-mean pipeline.

Three operations:

- `orthogonal_procrustes(A, B)`: closed-form rotation R minimising ||AR − B||_F.
- `generalized_procrustes(mats)`: iterative GPA — rotate each matrix onto a
  consensus mean until the mean stabilises.
- `mcca_basket(Zs, D)`: Carroll/Kettenring MAX-VAR Generalized CCA — the
  shared latent that maximises sum-of-variances across all views.

All three operate on already-whitened per-model PCA features (per-feature
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


def mcca_basket(Zs: list[np.ndarray], D: int, seed: int = 0) -> np.ndarray:
    """Carroll/Kettenring MAX-VAR Generalized CCA shared latent.

    Given per-model whitened features Z_m ∈ R^{N×D}, stack horizontally:
        C = [Z_1 | Z_2 | ... | Z_M] ∈ R^{N × MD}
    The MAX-VAR GCCA solution is the top-D left singular vectors of C,
    scaled by their singular values: S = U[:, :D] @ diag(Σ[:D]).

    Each Z_m must already be on comparable scale (per-feature z-scored).
    Returns S ∈ R^{N × D}; intended as a third basket source alongside
    the naive mean and the Procrustes-aligned mean.
    """
    from sklearn.utils.extmath import randomized_svd

    C = np.concatenate(Zs, axis=1).astype(np.float32)
    U, sv, _ = randomized_svd(C, n_components=D, random_state=seed)
    return (U * sv).astype(np.float32)
