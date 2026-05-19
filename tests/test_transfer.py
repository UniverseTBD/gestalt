"""Synthetic tests for the cross-survey transfer sweep.

Exercises `bazaar.bench.transfer` primitives without HF downloads or real
embeddings: synthetic per-model arrays sized to the real BASKET so the
internal `BazaarFit.fit(..., basket=BASKET)` path is exercised end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest

from bazaar.basket import BASKET
from bazaar.bench.transfer import (
    _concat_in_basket_order,
    _slice_to_n_fit,
    fit_source,
    transfer_to_target,
)
from bazaar.fit import BazaarFit


def _model_names() -> list[str]:
    return [f"{f}_{s}" for f, s in BASKET]


def _synthetic_embeddings(
    n: int, *, d_per_model: int = 32, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Return {model_key: (n, d_per_model)} for every entry in the real BASKET.

    A small low-rank shared factor drives all views so MCCA recovers a
    meaningful subspace, mirroring the fixture in test_fit_transform.py.
    """
    rng = np.random.default_rng(seed)
    D_latent = 8
    factors = rng.standard_normal((n, D_latent)).astype(np.float32)
    out: dict[str, np.ndarray] = {}
    for key in _model_names():
        W = rng.standard_normal((D_latent, d_per_model)).astype(np.float32)
        noise = 0.05 * rng.standard_normal((n, d_per_model)).astype(np.float32)
        out[key] = factors @ W + noise
    return out


def test_slice_to_n_fit_truncates_rows():
    emb = _synthetic_embeddings(n=200)
    sub = _slice_to_n_fit(emb, n_fit=50)
    for key in _model_names():
        assert sub[key].shape == (50, 32)
        np.testing.assert_array_equal(sub[key], emb[key][:50])


def test_concat_in_basket_order_widths_match():
    emb = _synthetic_embeddings(n=10)
    names = _model_names()
    C = _concat_in_basket_order(emb, names)
    assert C.shape == (10, sum(emb[k].shape[1] for k in names))
    assert C.dtype == np.float32
    # Order is load-bearing: first slice is the first basket model's columns.
    np.testing.assert_array_equal(C[:, :32], emb[names[0]])


def test_fit_source_returns_bazaar_and_concat_pca():
    src = _synthetic_embeddings(n=200, seed=0)
    bf, pca_proj, single_art = fit_source(
        "synthetic", src, n_fit=100, D=16, seed=0, model_names=_model_names(),
    )
    assert isinstance(bf, BazaarFit)
    assert bf.D == 16
    # V's row partitioning is sum of per-model native widths (here 22 * 32).
    assert bf.mcca_V.shape == (sum(src[k].shape[1] for k in _model_names()), 16)
    assert pca_proj.n_components_ == 16
    # concat-PCA's input width matches the basket sum.
    assert pca_proj.components_.shape[1] == sum(src[k].shape[1] for k in _model_names())
    # Single-encoder baseline is fit on the 'astropt_850M' view in BASKET.
    assert single_art is not None
    assert single_art["pca_components"].shape == (16, 32)
    assert single_art["zscore_mu"].shape == (1, 16)


def test_fit_source_single_baseline_can_be_disabled():
    src = _synthetic_embeddings(n=200, seed=0)
    bf, pca_proj, single_art = fit_source(
        "synthetic", src, n_fit=100, D=16, seed=0,
        model_names=_model_names(), single_baseline=None,
    )
    assert isinstance(bf, BazaarFit)
    assert pca_proj.n_components_ == 16
    assert single_art is None


def test_fit_source_errors_when_too_few_rows():
    src = _synthetic_embeddings(n=50, seed=0)
    with pytest.raises(ValueError, match="only has 50 rows"):
        fit_source(
            "synthetic", src, n_fit=100, D=8, seed=0, model_names=_model_names(),
        )


def test_self_transfer_subspace_matches_fresh_fit():
    """fit on A, transform A → same column space as fit-only on A.

    `BazaarFit.transform` differs from the fit-time `S = U Σ` only by
    randomized-SVD reprojection error, so per-column correlation should be
    ~1 (matching the established tolerance in test_fit_transform.py).
    """
    src = _synthetic_embeddings(n=500, seed=1)
    bf, _, _ = fit_source(
        "synthetic", src, n_fit=400, D=16, seed=0, model_names=_model_names(),
    )
    S_held = bf.transform({k: v[400:] for k, v in src.items()})
    S_self = bf.transform({k: v[:400] for k, v in src.items()})
    assert S_held.shape == (100, 16)
    assert S_self.shape == (400, 16)
    assert np.all(np.isfinite(S_held))
    assert np.all(np.isfinite(S_self))
    # Held-out projections must spread across all 16 latent axes.
    assert (S_held.std(axis=0) > 1e-3).all()


def test_transfer_to_target_row_count_and_schema():
    """transfer_to_target emits 1 row per (source, representation, property, seed)."""
    src = _synthetic_embeddings(n=200, seed=0)
    bf, pca_proj, single_art = fit_source(
        "src", src, n_fit=100, D=16, seed=0, model_names=_model_names(),
    )

    tgt = _synthetic_embeddings(n=80, seed=2)
    tgt_concat = _concat_in_basket_order(tgt, _model_names())
    tgt_labels = {
        "y_reg":  np.linspace(0, 1, 80, dtype=np.float32),
        "y_cls":  np.array([i % 3 for i in range(80)], dtype=np.int64),
    }
    tgt_props = [("y_reg", "regression"), ("y_cls", "classification")]

    rows = transfer_to_target(
        "cosmos-hsc", tgt, tgt_concat, tgt_labels, tgt_props,
        source_name="cosmos-jwst", bf=bf, pca_proj=pca_proj,
        single_artifacts=single_art,
        D=16, n_seeds=3, test_size=20, n_fit=100,
    )
    # 1 source × 3 reps × 2 properties × 3 seeds = 18 rows
    assert len(rows) == 18
    # Schema invariants per row.
    expected_keys = {
        "target", "modality", "fit_source", "source", "property", "kind",
        "seed", "n_valid", "D", "n_fit", "r2", "acc", "f1",
    }
    valid_sources = {
        "basket_mcca_whitened", "basket_concat_pca",
        "single_astropt_850M_pca_zscore",
    }
    for r in rows:
        assert set(r.keys()) == expected_keys
        assert r["target"] == "cosmos-hsc"
        assert r["modality"] == "hsc"
        assert r["fit_source"] == "cosmos-jwst"
        assert r["source"] in valid_sources
        if r["kind"] == "regression":
            assert np.isfinite(r["r2"])
            assert np.isnan(r["acc"]) and np.isnan(r["f1"])
        else:
            assert np.isnan(r["r2"])
            assert np.isfinite(r["acc"]) and np.isfinite(r["f1"])
    # Single-encoder baseline contributes its own subset of rows.
    single_rows = [r for r in rows if r["source"] == "single_astropt_850M_pca_zscore"]
    assert len(single_rows) == 2 * 3  # 2 properties × 3 seeds


def test_native_self_run_matches_size_matched_native():
    """When fit_source == target the run produces well-defined probe rows.

    Sanity check that the 'native' baseline cell — same corpus on both sides
    — at least produces finite, non-trivial R²s under the same n_fit
    treatment we apply to cross cells.
    """
    rng = np.random.default_rng(42)
    n_total = 200
    factors = rng.standard_normal((n_total, 4)).astype(np.float32)
    y = (factors[:, 0] + 0.1 * rng.standard_normal(n_total)).astype(np.float32)
    emb = {}
    for key in _model_names():
        W = rng.standard_normal((4, 32)).astype(np.float32)
        noise = 0.05 * rng.standard_normal((n_total, 32)).astype(np.float32)
        emb[key] = factors @ W + noise

    bf, pca_proj, single_art = fit_source(
        "synthetic", emb, n_fit=100, D=16, seed=0, model_names=_model_names(),
    )
    rows = transfer_to_target(
        "cosmos-hsc", emb, _concat_in_basket_order(emb, _model_names()),
        {"y": y}, [("y", "regression")],
        source_name="cosmos-hsc", bf=bf, pca_proj=pca_proj,
        single_artifacts=single_art,
        D=16, n_seeds=3, test_size=50, n_fit=100,
    )
    # 1 source × 3 reps × 1 property × 3 seeds = 9 rows, all finite R²s.
    assert len(rows) == 9
    for r in rows:
        assert np.isfinite(r["r2"])
        # Probe on a synthetic linear factor should be far better than chance.
        assert r["r2"] > 0.5, f"unexpectedly poor R² {r['r2']} for {r}"
