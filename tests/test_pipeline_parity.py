"""Parity between BazaarFit-driven MCCA and legacy `mcca_basket`.

`run_modality` previously computed the MCCA basket source as
`mcca_basket(Zs, D, seed)` — returning `U * sv` from the randomized SVD of
`C = [Z_1 | ... | Z_M]`. The refactor computes it as `fit.transform(E)` —
returning `C_recomputed @ V` where `V = Vh.T` from the same SVD. The two
agree exactly when randomized SVD is exact; in practice they drift by
randomized-SVD reprojection error (~1e-4 on float32). This file asserts
the drift is small enough that downstream linear-probe R² is preserved.
"""
from __future__ import annotations

import numpy as np

from bazaar.align import mcca_basket
from bazaar.fit import BazaarFit
from bazaar.pipeline import pca_zscore_fit
from bazaar.probe import run_probe


def _synthetic_basket(
    n: int = 4000, d: int = 64, D: int = 16, n_models: int = 4, seed: int = 0,
):
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n, D)).astype(np.float32)
    basket = [("fam", f"m{i:02d}") for i in range(n_models)]
    embeddings: dict[str, np.ndarray] = {}
    for fam, size in basket:
        W = rng.standard_normal((D, d)).astype(np.float32)
        noise = 0.05 * rng.standard_normal((n, d)).astype(np.float32)
        embeddings[f"{fam}_{size}"] = factors @ W + noise
    return embeddings, basket, factors


def test_mcca_basket_close_to_fit_transform():
    """Legacy U*sv and refactored C@V differ only by randomized-SVD error."""
    embeddings, basket, _ = _synthetic_basket()
    fit = BazaarFit.fit(embeddings, basket=basket, D=16, seed=0)

    Zs = [pca_zscore_fit(embeddings[f"{f}_{s}"], D=16, seed=0)[0] for f, s in basket]
    S_legacy = mcca_basket(Zs, D=16, seed=0)
    S_new = fit.transform(embeddings)

    # Both live in the same column space up to randomized-SVD residual.
    # Per-column |corr| must be ~1 (sign-free).
    for k in range(S_legacy.shape[1]):
        c = np.corrcoef(S_legacy[:, k], S_new[:, k])[0, 1]
        assert abs(c) > 0.999, f"column {k}: |corr|={abs(c):.4f}"


def test_probe_r2_parity_within_tolerance():
    """Linear-probe R² on the new MCCA basket source matches legacy to <1e-3."""
    embeddings, basket, factors = _synthetic_basket()
    fit = BazaarFit.fit(embeddings, basket=basket, D=16, seed=0)

    Zs = [pca_zscore_fit(embeddings[f"{f}_{s}"], D=16, seed=0)[0] for f, s in basket]
    S_legacy = mcca_basket(Zs, D=16, seed=0)
    S_new = fit.transform(embeddings)

    # Synthetic regression target: a known linear combination of factors.
    rng = np.random.default_rng(123)
    beta = rng.standard_normal(factors.shape[1]).astype(np.float32)
    y = (factors @ beta + 0.1 * rng.standard_normal(factors.shape[0])).astype(np.float32)

    r2_legacy = run_probe(S_legacy, y, test_size=400, random_state=0)
    r2_new = run_probe(S_new, y, test_size=400, random_state=0)
    assert abs(r2_legacy - r2_new) < 1e-3, (
        f"R² drift {abs(r2_legacy - r2_new):.4e} exceeds tolerance"
    )
