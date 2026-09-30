"""Tests for `gestalt bench probes` (PU Fig 4 cosine matrix replication)."""

from __future__ import annotations

import numpy as np

from gestalt.bench._probe_coeffs import fit_probe_coeffs
from gestalt.bench.probe_geometry import (
    PROPERTIES,
    _three_probe_cos_matrix,
    run_probes,
)


def _synthetic_basket_with_known_directions(
    n: int = 4000,
    d: int = 64,
    D: int = 8,
    n_models: int = 4,
    seed: int = 0,
):
    """Basket where (z, M★, sSFR) sit on three known factor directions.

    factor[:, 0] = z; factor[:, 1] = M★; factor[:, 2] = sSFR.
    We deliberately correlate factor 1 and factor 2 mildly negatively so
    cos(M★, sSFR) should come out modestly negative — the PU §3.1
    main-sequence sign.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, D)).astype(np.float32)
    # Make M★ and sSFR mildly anti-correlated by giving them a shared
    # negative component.
    mass = base[:, 1].copy()
    ssfr = -0.4 * mass + np.sqrt(1 - 0.4**2) * base[:, 2]
    factors = base.copy()
    factors[:, 2] = ssfr.astype(np.float32)
    labels = {
        "redshift": factors[:, 0].copy(),
        "mass": factors[:, 1].copy(),
        "sSFR": factors[:, 2].copy(),
    }

    basket = [("fam", f"m{i:02d}") for i in range(n_models)]
    embeddings = {}
    for fam, size in basket:
        W = rng.standard_normal((D, d)).astype(np.float32)
        noise = 0.05 * rng.standard_normal((n, d)).astype(np.float32)
        embeddings[f"{fam}_{size}"] = factors @ W + noise
    return embeddings, basket, labels


def test_fit_probe_coeffs_recovers_known_direction():
    """If y is a column of X exactly, the un-scaled w should point along
    that column with cos ≈ 1.
    """
    rng = np.random.default_rng(0)
    N, D = 500, 6
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = X[:, 2]
    out = fit_probe_coeffs(X, y)
    e = np.zeros(D, dtype=np.float32)
    e[2] = 1.0
    w = out["w"]
    cos = float(w @ e / (np.linalg.norm(w) * np.linalg.norm(e) + 1e-12))
    assert cos > 0.999
    assert out["n_valid"] == N


def test_three_probe_cos_matrix_recovers_orthogonal_directions():
    """When (z, M★, sSFR) sit on three orthogonal columns, cosines are
    diag(1) + off-diag ≈ 0.
    """
    rng = np.random.default_rng(1)
    N, D = 1500, 8
    Z = rng.standard_normal((N, D)).astype(np.float32)
    labels = {"redshift": Z[:, 0], "mass": Z[:, 1], "sSFR": Z[:, 2]}
    C = _three_probe_cos_matrix(Z, labels)
    assert C.shape == (3, 3)
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-3)
    # All off-diagonals near zero (within sampling noise on N=1500).
    off_diag = C[~np.eye(3, dtype=bool)]
    assert np.abs(off_diag).max() < 0.1


def test_three_probe_cos_matrix_recovers_negative_ssfr_mass():
    """Deliberate anti-correlation in the labels → cos(M★, sSFR) < 0."""
    rng = np.random.default_rng(2)
    N, D = 2000, 6
    Z = rng.standard_normal((N, D)).astype(np.float32)
    # Make M★ and sSFR labels anti-correlated.
    m = Z[:, 1].copy()
    s = -0.6 * m + np.sqrt(1 - 0.36) * Z[:, 2]
    labels = {"redshift": Z[:, 0], "mass": m, "sSFR": s.astype(np.float32)}
    C = _three_probe_cos_matrix(Z, labels)
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-3)
    # cos(M★, sSFR) should recover ≈ -0.6.
    assert C[1, 2] < -0.4
    assert C[1, 2] > -0.8


def test_run_probes_schema_and_basket_avg_consistency():
    """End-to-end on the synthetic basket: emit Gestalt + 4 per-model + basket-avg
    rows; basket_avg must equal elementwise mean of the per-model matrices.
    """
    embeddings, basket, labels = _synthetic_basket_with_known_directions(
        n=2000,
        d=32,
        D=8,
        n_models=4,
        seed=3,
    )
    rows = run_probes(
        "hsc",
        embeddings,
        labels,
        basket,
        D=8,
        seed=0,
    )
    # Schema: each source emits 9 rows (3×3 matrix).
    sources = {r["source"] for r in rows}
    assert "basket_mcca_whitened" in sources
    assert "basket_avg" in sources
    n_singles = sum(1 for s in sources if s.startswith("single_"))
    assert n_singles == len(basket)
    # 1 (gestalt) + len(basket) (per-model) + 1 (avg) → 9 rows each.
    expected = (1 + n_singles + 1) * 9
    assert len(rows) == expected
    # Every cosine is in [-1, 1]; diagonals are exactly 1.
    for r in rows:
        assert -1.001 <= r["cos"] <= 1.001
        if r["prop_i"] == r["prop_j"]:
            assert abs(r["cos"] - 1.0) < 1e-3

    # basket_avg is the elementwise mean of the per-model matrices.
    def _matrix(source: str) -> np.ndarray:
        m = np.zeros((3, 3), dtype=np.float32)
        for r in rows:
            if r["source"] != source:
                continue
            i = PROPERTIES.index(r["prop_i"])
            j = PROPERTIES.index(r["prop_j"])
            m[i, j] = r["cos"]
        return m

    avg = _matrix("basket_avg")
    singles = np.stack([_matrix(s) for s in sources if s.startswith("single_")])
    np.testing.assert_allclose(avg, singles.mean(axis=0), atol=1e-5)


def test_run_probes_recovers_negative_ssfr_mass_off_diagonal():
    """With label-level anti-correlation between M★ and sSFR, Gestalt's
    off-diagonal cos(M★, sSFR) should come out clearly negative.
    """
    embeddings, basket, labels = _synthetic_basket_with_known_directions(
        n=3000,
        d=32,
        D=8,
        n_models=4,
        seed=4,
    )
    rows = run_probes(
        "hsc",
        embeddings,
        labels,
        basket,
        D=8,
        seed=0,
    )
    gestalt_rows = [
        r
        for r in rows
        if r["source"] == "basket_mcca_whitened" and r["prop_i"] == "mass" and r["prop_j"] == "sSFR"
    ]
    assert len(gestalt_rows) == 1
    cos = gestalt_rows[0]["cos"]
    # Anti-correlation between labels (≈ -0.4) is preserved through the
    # basket; allow generous slack for MCCA / probe sampling noise.
    assert cos < -0.1


def test_run_probes_row_count_mismatch_errors():
    embeddings, basket, labels = _synthetic_basket_with_known_directions(
        n=500,
        d=16,
        D=4,
        n_models=3,
        seed=5,
    )
    bad_labels = {**labels, "redshift": np.zeros(100, dtype=np.float32)}
    try:
        run_probes("hsc", embeddings, bad_labels, basket, D=4, seed=0)
    except RuntimeError as e:
        assert "row mismatch" in str(e)
        return
    raise AssertionError("expected RuntimeError for row mismatch")
