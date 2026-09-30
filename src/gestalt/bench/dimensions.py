"""Per-dimension covariate decomposition — `gestalt bench dims`.

Asks "what is each Gestalt latent dimension actually encoding?" by regressing
every coordinate of the D-dim shared latent `S = fit.transform(embeddings)`
against a wide catalogue of physical, photometric, morphological, systematic,
and positional covariates. Output is a long-form `(modality, dim, covariate,
covariate_group, n_valid, r2)` parquet; downstream plotting renders it as a
(covariate × dim) R² heatmap per modality.

Compared with `bench cosmos` (which probes the full subspace against three
labels), this sweep is per-coordinate: it surfaces which dims carry redshift,
which carry magnitude/depth, which carry survey-footprint signal (RA/Dec),
and so on. A clean basket fit should spread informational load — if RA/Dec or
per-band SNR dominate the argmax-per-dim distribution, V has picked up
non-physical structure as a latent axis and we want to know.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gestalt.basket import BASKET, load_embeddings
from gestalt.bench.cosmosweb import catalog_pass_wide
from gestalt.fit import GestaltFit


def _pairwise_r2(
    S: np.ndarray,
    X: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Pearson r² between every column of S and every column of X.

    Parameters
    ----------
    S      : (N, D)  — Gestalt shared latent. Assumed dense.
    X      : (N, C)  — covariate matrix, may contain NaNs.
    valid  : (N, C) bool — `~isnan(X)` precomputed.

    Returns
    -------
    r2      : (D, C) float32 — squared Pearson correlation per (dim, covariate).
    n_valid : (C,)  int64    — rows used per covariate.
    """
    N, D = S.shape
    _, C = X.shape

    n_valid = valid.sum(axis=0).astype(np.int64)
    if (n_valid < 3).any():
        bad = np.where(n_valid < 3)[0]
        raise ValueError(f"covariate columns {bad.tolist()} have <3 finite rows")

    # Z-score S once (constant across covariates).
    S_mean = S.mean(axis=0, keepdims=True)
    S_sd = S.std(axis=0, keepdims=True) + 1e-8
    S_z = (S - S_mean) / S_sd

    r2 = np.empty((D, C), dtype=np.float32)
    for c in range(C):
        m = valid[:, c]
        if m.all():
            x = X[:, c]
            x_z = (x - x.mean()) / (x.std() + 1e-8)
            # Correlation column-wise: corr_d = (S_z[:, d] @ x_z) / N.
            r = (S_z.T @ x_z) / N
        else:
            n = int(m.sum())
            x = X[m, c]
            x_z = (x - x.mean()) / (x.std() + 1e-8)
            # Re-zscore the masked S slice — z-stats change when we drop rows.
            S_m = S[m]
            S_mz = (S_m - S_m.mean(axis=0, keepdims=True)) / (S_m.std(axis=0, keepdims=True) + 1e-8)
            r = (S_mz.T @ x_z) / n
        r2[:, c] = (r * r).astype(np.float32)
    return r2, n_valid


def run_dimensions(
    telescope: str,
    embeddings: dict[str, np.ndarray],
    covariates: dict[str, np.ndarray],
    groups: dict[str, str],
    basket: list[tuple[str, str]],
    *,
    D: int,
    seed: int = 0,
) -> list[dict]:
    """Fit Gestalt at D on this modality; emit per-(dim, covariate) R² rows."""
    n_emb = next(iter(embeddings.values())).shape[0]
    n_cov = next(iter(covariates.values())).shape[0]
    if n_cov != n_emb:
        raise RuntimeError(
            f"row mismatch: embeddings have {n_emb} rows, "
            f"covariates have {n_cov} — streaming order may have drifted."
        )

    print(f"\n[gestalt.dims] === {telescope.upper()} (D={D}) ===")
    print(f"[gestalt.dims] GestaltFit.fit on {n_emb} rows × {len(embeddings)} models")
    fit = GestaltFit.fit(embeddings, basket=basket, D=D, seed=seed)
    S = fit.transform(embeddings).astype(np.float32, copy=False)
    print(f"[gestalt.dims] shared latent S shape={S.shape}")

    cov_names = list(covariates.keys())
    X = np.stack([covariates[n] for n in cov_names], axis=1)  # (N, C)
    valid = np.isfinite(X)

    r2, n_valid = _pairwise_r2(S, X, valid)
    print(f"[gestalt.dims] computed R² matrix {r2.shape} (dim × covariate)")

    rows: list[dict] = []
    D_eff = r2.shape[0]
    for c_idx, c_name in enumerate(cov_names):
        for d in range(D_eff):
            rows.append(
                dict(
                    modality=telescope,
                    dim=int(d),
                    covariate=c_name,
                    covariate_group=groups[c_name],
                    n_valid=int(n_valid[c_idx]),
                    r2=float(r2[d, c_idx]),
                    D=int(D),
                )
            )

    _print_per_modality_summary(rows, telescope)
    return rows


def _print_per_modality_summary(rows: list[dict], telescope: str) -> None:
    """One-line-per-covariate summary: max R² and its argmax dim."""
    by_cov: dict[str, list[dict]] = {}
    for r in rows:
        if r["modality"] != telescope:
            continue
        by_cov.setdefault(r["covariate"], []).append(r)
    print(f"[gestalt.dims] {telescope.upper()} argmax-dim per covariate (top 10 by max R²):")
    summary = []
    for cov, ents in by_cov.items():
        best = max(ents, key=lambda r: r["r2"])
        summary.append((cov, best["covariate_group"], best["r2"], best["dim"]))
    summary.sort(key=lambda t: t[2], reverse=True)
    for cov, grp, r2, dim in summary[:10]:
        print(f"  {cov:24s} [{grp:11s}]  max_r2={r2:.4f}  argmax_dim={dim:4d}")


def run_dimensions_cosmos(
    *,
    D: int,
    n_use: int,
    emb_dir: Path,
    telescopes: tuple[str, ...] = ("hsc", "jwst"),
    seed: int = 0,
) -> list[dict]:
    """Top-level entry: covariate pass once, run_dimensions per telescope."""
    covariates, groups = catalog_pass_wide(n_use)
    print(
        f"[gestalt.dims] {len(covariates)} covariates over {n_use} rows "
        f"({sorted(set(groups.values()))})"
    )

    all_rows: list[dict] = []
    for tele in telescopes:
        embeddings = load_embeddings(BASKET, tele, emb_dir, n_use=n_use)
        all_rows.extend(
            run_dimensions(
                tele,
                embeddings,
                covariates,
                groups,
                BASKET,
                D=D,
                seed=seed,
            )
        )
        del embeddings
    return all_rows


__all__ = [
    "run_dimensions",
    "run_dimensions_cosmos",
]
