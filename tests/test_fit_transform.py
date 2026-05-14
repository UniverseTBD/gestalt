"""Tests for BazaarFit fit/transform/save/load round-trips."""
from __future__ import annotations

import numpy as np
import pytest

from bazaar.align import mcca_fit
from bazaar.fit import BazaarFit
from bazaar.whiten import pca_zscore_fit


def _synthetic_basket(
    n: int = 4000, d: int = 64, D: int = 16, n_models: int = 4, seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    """A small basket with low-rank shared structure across views.

    Random Gaussian factors drive each per-model embedding via a per-model
    loading matrix, so PCA recovers the same subspace on each view.
    """
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n, D)).astype(np.float32)
    basket = [("fam", f"m{i:02d}") for i in range(n_models)]
    embeddings: dict[str, np.ndarray] = {}
    for fam, size in basket:
        W = rng.standard_normal((D, d)).astype(np.float32)
        noise = 0.05 * rng.standard_normal((n, d)).astype(np.float32)
        embeddings[f"{fam}_{size}"] = factors @ W + noise
    return embeddings, basket


def test_fit_is_full_rank_per_model():
    """Per-model PCA goes to native rank min(d_in, N) — no truncation to D."""
    embeddings, basket = _synthetic_basket(n=4000, d=64, D=16)
    fit = BazaarFit.fit(embeddings, basket=basket, D=16, seed=0)
    # Each per-model PCA stores 64 components (d_in), not D=16.
    for art in fit.pca.values():
        assert art["pca_components"].shape == (64, 64)
        assert art["zscore_mu"].shape == (1, 64)
    # V's row dim is the sum of per-model widths (here 4 * 64), not M*D.
    assert fit.mcca_V.shape == (4 * 64, 16)


def test_fit_stores_same_V_as_direct_mcca():
    """BazaarFit.fit's V matches `mcca_fit` invoked on the same internal Zs."""
    embeddings, basket = _synthetic_basket()
    fit = BazaarFit.fit(embeddings, basket=basket, D=16, seed=0)

    # Reconstruct the exact fit-time Zs (full-rank PCA + zscore per view).
    Zs_ref = [
        pca_zscore_fit(embeddings[f"{f}_{s}"], D=embeddings[f"{f}_{s}"].shape[1],
                       seed=0)[0]
        for f, s in basket
    ]
    V_ref, _ = mcca_fit(Zs_ref, D=16, seed=0)
    np.testing.assert_array_equal(fit.mcca_V, V_ref)


def test_transform_on_fit_data_recovers_subspace():
    """`fit.transform(fit_E)` correlates ~1 with the fit-time S column-wise.

    Not bit-identical because `pca_zscore_transform` is a lossy reprojection
    (randomized-SVD approximation), but the column space is preserved.
    """
    embeddings, basket = _synthetic_basket()
    fit = BazaarFit.fit(embeddings, basket=basket, D=16, seed=0)

    Zs_ref = [
        pca_zscore_fit(embeddings[f"{f}_{s}"], D=embeddings[f"{f}_{s}"].shape[1],
                       seed=0)[0]
        for f, s in basket
    ]
    _, S_ref = mcca_fit(Zs_ref, D=16, seed=0)
    S_tr = fit.transform(embeddings)

    for k in range(S_ref.shape[1]):
        c = np.corrcoef(S_ref[:, k], S_tr[:, k])[0, 1]
        assert abs(c) > 0.999, f"column {k}: |corr|={abs(c):.4f}"


def test_save_load_round_trip(tmp_path):
    embeddings, basket = _synthetic_basket()
    fit = BazaarFit.fit(embeddings, basket=basket, D=16, seed=0)
    fit.save(tmp_path / "fit")
    reloaded = BazaarFit.load(tmp_path / "fit")

    assert reloaded.D == fit.D
    assert reloaded.seed == fit.seed
    assert reloaded.basket == fit.basket
    np.testing.assert_array_equal(reloaded.mcca_V, fit.mcca_V)
    for key, art in fit.pca.items():
        for ak, av in art.items():
            np.testing.assert_array_equal(reloaded.pca[key][ak], av)

    np.testing.assert_allclose(
        reloaded.transform(embeddings), fit.transform(embeddings), atol=0,
    )


def test_held_out_transform_uses_same_basis(tmp_path):
    """Held-out rows fed through transform live in the same subspace."""
    embeddings, basket = _synthetic_basket(n=5000)
    train = {k: v[:4000] for k, v in embeddings.items()}
    test = {k: v[4000:] for k, v in embeddings.items()}

    fit = BazaarFit.fit(train, basket=basket, D=16, seed=0)
    S_test = fit.transform(test)

    assert S_test.shape == (1000, 16)
    assert np.all(np.isfinite(S_test))
    assert (S_test.std(axis=0) > 1e-2).all()


def test_mismatched_basket_raises(tmp_path):
    embeddings, basket = _synthetic_basket()
    fit = BazaarFit.fit(embeddings, basket=basket, D=16, seed=0)
    bad = {k: v for k, v in embeddings.items() if k != f"{basket[0][0]}_{basket[0][1]}"}
    with pytest.raises(KeyError):
        fit.transform(bad)


def test_legacy_schema_rejected(tmp_path):
    """Fits saved under older schema versions error with a clear message."""
    import json

    embeddings, basket = _synthetic_basket()
    fit = BazaarFit.fit(embeddings, basket=basket, D=16, seed=0)
    fit.save(tmp_path / "legacy")
    meta_path = tmp_path / "legacy" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = 1
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="Unsupported schema_version"):
        BazaarFit.load(tmp_path / "legacy")
