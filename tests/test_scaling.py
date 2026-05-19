"""Tests for the basket-pruning sweep (`bazaar bench scaling`)."""
from __future__ import annotations

import numpy as np

from bazaar.basket import BASKET
from bazaar.bench.scaling import (
    build_basket_sources_subset,
    iter_full_basket,
    iter_one_per_family_subsets,
    iter_random_subsets,
    precompute_full_whitened,
)


def _names(members) -> list[str]:
    return [f"{f}_{s}" for f, s in members]


def test_iter_random_subsets_size_and_count():
    subsets = list(iter_random_subsets(BASKET, k=4, n_replicates=5, seed=0))
    assert len(subsets) == 5
    for subset_id, members in subsets:
        assert isinstance(subset_id, int)
        assert len(members) == 4
        # All members must come from the basket.
        for m in members:
            assert m in BASKET
        # No duplicate models within a subset.
        assert len(set(_names(members))) == 4


def test_iter_random_subsets_seed_determinism():
    a = [tuple(_names(m)) for _, m in iter_random_subsets(BASKET, 8, 5, seed=42)]
    b = [tuple(_names(m)) for _, m in iter_random_subsets(BASKET, 8, 5, seed=42)]
    c = [tuple(_names(m)) for _, m in iter_random_subsets(BASKET, 8, 5, seed=7)]
    assert a == b
    assert a != c


def test_iter_random_subsets_k_too_large_yields_nothing():
    assert list(iter_random_subsets(BASKET, k=99, n_replicates=3, seed=0)) == []


def test_iter_one_per_family_unique_families():
    # 8 families exist in BASKET; at k=8 each subset has one per family.
    families_present = sorted({f for f, _ in BASKET})
    for _, members in iter_one_per_family_subsets(BASKET, k=8, n_replicates=5, seed=0):
        fams = [f for f, _ in members]
        assert len(fams) == 8
        assert sorted(fams) == families_present  # exactly one per family

    # At k=4, each subset has exactly 4 distinct families.
    for _, members in iter_one_per_family_subsets(BASKET, k=4, n_replicates=5, seed=0):
        fams = [f for f, _ in members]
        assert len(fams) == 4
        assert len(set(fams)) == 4


def test_iter_one_per_family_skipped_above_n_families():
    # 8 families; k=10 cannot pick 10 distinct families.
    assert list(iter_one_per_family_subsets(BASKET, k=10, n_replicates=3, seed=0)) == []


def test_iter_full_basket_singleton():
    subs = list(iter_full_basket(BASKET))
    assert len(subs) == 1
    sid, members = subs[0]
    assert sid == 0
    assert members == list(BASKET)


def _synthetic_basket():
    """A small heterogeneous basket: 3 families × 2 sizes, varying widths."""
    rng = np.random.default_rng(0)
    n, D_shared = 300, 8
    factors = rng.standard_normal((n, D_shared)).astype(np.float32)
    basket = [
        ("famA", "s1"), ("famA", "s2"),
        ("famB", "s1"), ("famB", "s2"),
        ("famC", "s1"), ("famC", "s2"),
    ]
    widths = [32, 48, 24, 40, 56, 36]
    embeddings: dict[str, np.ndarray] = {}
    for (fam, size), d in zip(basket, widths):
        W = rng.standard_normal((D_shared, d)).astype(np.float32)
        noise = 0.05 * rng.standard_normal((n, d)).astype(np.float32)
        embeddings[f"{fam}_{size}"] = factors @ W + noise
    return embeddings, basket


def test_build_basket_sources_subset_shapes_and_finite():
    embeddings, basket = _synthetic_basket()
    names = [f"{f}_{s}" for f, s in basket]
    Zs_white = precompute_full_whitened(embeddings, names)

    # Random subset of 3 of 6.
    sources = build_basket_sources_subset(
        Zs_white, subset_names=names[:3], D=16,
        log_prefix="[test]",
    )
    by_name = dict(sources)
    assert set(by_name) == {"basket_mcca_whitened"}
    B = by_name["basket_mcca_whitened"]
    assert B.shape == (300, 16)
    assert np.all(np.isfinite(B))


def test_build_basket_sources_subset_full_basket_matches_runner():
    """When the subset is the full basket, MCCA output equals `_runner`'s."""
    from bazaar.bench._runner import build_basket_sources

    embeddings, basket = _synthetic_basket()
    names = [f"{f}_{s}" for f, s in basket]
    Zs_white = precompute_full_whitened(embeddings, names)

    runner_sources = dict(build_basket_sources(embeddings, names, D=16))
    subset_sources = dict(build_basket_sources_subset(
        Zs_white, subset_names=names, D=16, log_prefix="[test]",
    ))
    # MCCA output must be bit-for-bit identical (same whitening, same seed).
    np.testing.assert_array_equal(
        subset_sources["basket_mcca_whitened"],
        runner_sources["basket_mcca_whitened"],
    )
