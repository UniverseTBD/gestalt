"""Tests for `gestalt bench dims` (per-dimension covariate regression).

Strategy: build a synthetic basket with low-rank shared structure driven by a
known set of factors. Then construct covariates that have known linear
relationships to those factors. After fitting Gestalt, the resulting S spans
the factor subspace (up to rotation), so each covariate's max R² over dims
should be close to 1.0; unrelated noise covariates should have low max R².
"""

from __future__ import annotations

import numpy as np

from gestalt.bench.dimensions import _pairwise_r2, run_dimensions


def _synthetic_basket(
    n: int = 4000,
    d: int = 64,
    D: int = 8,
    n_models: int = 4,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str]], np.ndarray]:
    """A small basket with low-rank shared structure across views.

    Returns (embeddings, basket, factors). `factors` is the latent (N, D)
    factor matrix that drives every view, so covariates correlated with
    individual factor columns will be recoverable from the Gestalt S.
    """
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n, D)).astype(np.float32)
    basket = [("fam", f"m{i:02d}") for i in range(n_models)]
    embeddings: dict[str, np.ndarray] = {}
    for fam, size in basket:
        W = rng.standard_normal((D, d)).astype(np.float32)
        noise = 0.05 * rng.standard_normal((n, d)).astype(np.float32)
        embeddings[f"{fam}_{size}"] = factors @ W + noise
    return embeddings, basket, factors


def test_pairwise_r2_shapes_and_range():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((500, 16)).astype(np.float32)
    X = rng.standard_normal((500, 7)).astype(np.float32)
    valid = np.isfinite(X)
    r2, n_valid = _pairwise_r2(S, X, valid)
    assert r2.shape == (16, 7)
    assert n_valid.shape == (7,)
    assert (r2 >= 0).all() and (r2 <= 1).all()
    assert (n_valid == 500).all()


def test_pairwise_r2_perfect_recovery_on_linear_signal():
    """If a covariate equals one column of S exactly, r² for that pair is 1."""
    rng = np.random.default_rng(1)
    N, D = 1000, 8
    S = rng.standard_normal((N, D)).astype(np.float32)
    X = np.stack([S[:, 0], S[:, 3], rng.standard_normal(N)], axis=1).astype(np.float32)
    r2, _ = _pairwise_r2(S, X, np.isfinite(X))
    # Cov 0 perfectly aligns with dim 0; cov 1 with dim 3; cov 2 is independent.
    assert r2[0, 0] > 0.999
    assert r2[3, 1] > 0.999
    # Other entries for that covariate should be small (orthogonal Gaussian dims).
    assert r2[1, 0] < 0.05
    assert r2[2, 2] < 0.05


def test_pairwise_r2_handles_nan_mask():
    rng = np.random.default_rng(2)
    N, D = 500, 4
    S = rng.standard_normal((N, D)).astype(np.float32)
    # Build a covariate equal to S[:, 1] with 30% NaNs.
    x = S[:, 1].copy()
    mask = rng.random(N) < 0.3
    x_nan = x.copy()
    x_nan[mask] = np.nan
    X = x_nan[:, None]
    valid = np.isfinite(X)
    r2, n_valid = _pairwise_r2(S, X, valid)
    assert r2.shape == (D, 1)
    assert n_valid[0] == (~mask).sum()
    # Still perfectly correlated on the unmasked rows.
    assert r2[1, 0] > 0.99


def test_run_dimensions_recovers_factor_signal():
    """End-to-end on the synthetic basket: covariate = factor_k → max r² ≈ 1."""
    embeddings, basket, factors = _synthetic_basket(
        n=2000,
        d=48,
        D=8,
        n_models=4,
        seed=3,
    )
    rng = np.random.default_rng(4)
    # Two covariates that linearly track factors, one pure-noise covariate.
    cov = {
        "track_factor0": factors[:, 0].copy(),
        "track_factor5": factors[:, 5].copy(),
        "pure_noise": rng.standard_normal(factors.shape[0]).astype(np.float32),
    }
    groups = {"track_factor0": "physics", "track_factor5": "physics", "pure_noise": "systematics"}

    rows = run_dimensions(
        "hsc",
        embeddings,
        cov,
        groups,
        basket,
        D=8,
        seed=0,
    )
    # Long-form schema sanity.
    assert {r["modality"] for r in rows} == {"hsc"}
    assert {r["covariate"] for r in rows} == set(cov.keys())
    assert all(0 <= r["r2"] <= 1 for r in rows)

    by_cov: dict[str, list[float]] = {}
    for r in rows:
        by_cov.setdefault(r["covariate"], []).append(r["r2"])
    # Each covariate emits one row per dim.
    for r2s in by_cov.values():
        assert len(r2s) == 8

    # MCCA rotates the factor subspace, so a single factor smears across
    # multiple Gestalt dims. The right "covariate is captured by S" test is
    # that the per-dim r² values sum to ≈ 1 (Gestalt's columns are orthogonal
    # post-SVD, so ∑ r²_d is the multivariate R² of cov ~ S). For pure noise
    # the sum should be near zero.
    assert sum(by_cov["track_factor0"]) > 0.85
    assert sum(by_cov["track_factor5"]) > 0.85
    # Pure noise's expected ∑r² ≈ D/N = 8/2000 = 0.004; tolerance up to 0.05.
    assert sum(by_cov["pure_noise"]) < 0.05


def test_run_dimensions_row_count_mismatch_errors():
    embeddings, basket, _ = _synthetic_basket(n=500, d=16, D=4, seed=5)
    # Covariates of mismatched length.
    cov = {"too_short": np.zeros(100, dtype=np.float32)}
    groups = {"too_short": "physics"}
    try:
        run_dimensions(
            "hsc",
            embeddings,
            cov,
            groups,
            basket,
            D=4,
            seed=0,
        )
    except RuntimeError as e:
        assert "row mismatch" in str(e)
        return
    raise AssertionError("expected a row-count mismatch error")
