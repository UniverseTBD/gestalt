"""End-to-end pipeline: PCA → align → average → probe.

For each modality (telescope):

  1. Load cached per-model embeddings (`bazaar.basket.ensure_embeddings_downloaded`).
  2. PCA each model to D components, then z-score each feature
     (`pca_and_zscore`).
  3. Compute three basket sources:
     - naive mean (no alignment)
     - Procrustes / GPA-aligned mean
     - MCCA shared latent
  4. Run the linear probe on each basket source and each single-model
     PCA feature for `n_seeds` random train/test splits.

Output is long-form: one row per (modality, property, seed, source).
"""
from __future__ import annotations

import numpy as np
from datasets import load_dataset
from sklearn.decomposition import PCA
from tqdm import tqdm

from bazaar.align import generalized_procrustes
from bazaar.basket import DATASET, load_embeddings
from bazaar.probe import run_probe

WHITEN_MODES = ("pca_zscore", "zscore")

# Physics parameter → dataset column for Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2.
CATALOG_COLUMNS = {
    "redshift": "lephare_photozs",
    "mag_g":    "mag_model_hsc-g",
    "mag_r":    "mag_model_hsc-r",
    "mass":     "lp_mass",
    "sSFR":     "lp_ssfr",
}

PROPERTIES = ["redshift", "mass", "sSFR"]


def pca_zscore_fit(
    E: np.ndarray, D: int, seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit per-model PCA + per-feature z-score on E.

    Returns
    -------
    Z         : (N, D_eff) float32 — whitened features on the fit data.
    artifacts : dict with keys
                  - pca_components : (D_eff, d_in)
                  - pca_mean       : (d_in,)
                  - zscore_mu      : (1, D_eff)
                  - zscore_sd      : (1, D_eff)
                Sufficient to reproduce Z via `pca_zscore_transform`.
    """
    D_eff = min(D, E.shape[1], E.shape[0])
    pca = PCA(n_components=D_eff, svd_solver="randomized", random_state=seed)
    Z = pca.fit_transform(E)
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True) + 1e-8
    artifacts = {
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean":       pca.mean_.astype(np.float32),
        "zscore_mu":      mu.astype(np.float32),
        "zscore_sd":      sd.astype(np.float32),
    }
    return ((Z - mu) / sd).astype(np.float32), artifacts


def pca_zscore_transform(E: np.ndarray, artifacts: dict[str, np.ndarray]) -> np.ndarray:
    """Apply saved PCA + z-score artifacts to (possibly new) embeddings."""
    E64 = E.astype(np.float32, copy=False)
    Z = (E64 - artifacts["pca_mean"]) @ artifacts["pca_components"].T
    return ((Z - artifacts["zscore_mu"]) / artifacts["zscore_sd"]).astype(np.float32)


def pca_and_zscore(E: np.ndarray, D: int, seed: int = 0) -> np.ndarray:
    """Back-compat: PCA on full E (randomized SVD), then per-feature z-score."""
    Z, _ = pca_zscore_fit(E, D=D, seed=seed)
    return Z


def zscore_fit(E: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit per-feature z-score on raw embeddings (no PCA, no dim reduction).

    The PCA-free counterpart of `pca_zscore_fit`, used by the `"zscore"`
    whiten mode in `BazaarFit` to ablate PCA out of the pipeline. The output
    keeps `E`'s column count, so downstream MCCA receives heterogeneous-
    width per-model views — fine for the SVD but breaks the naive mean
    and GPA paths, which assume matching shapes.
    """
    E32 = E.astype(np.float32, copy=False)
    mu = E32.mean(axis=0, keepdims=True)
    sd = E32.std(axis=0, keepdims=True) + 1e-8
    artifacts = {
        "zscore_mu": mu.astype(np.float32),
        "zscore_sd": sd.astype(np.float32),
    }
    return ((E32 - mu) / sd).astype(np.float32), artifacts


def zscore_transform(E: np.ndarray, artifacts: dict[str, np.ndarray]) -> np.ndarray:
    """Apply saved per-feature z-score stats to (possibly new) embeddings."""
    E32 = E.astype(np.float32, copy=False)
    return ((E32 - artifacts["zscore_mu"]) / artifacts["zscore_sd"]).astype(np.float32)


def catalog_pass(n_use: int) -> dict[str, np.ndarray]:
    """Stream the dataset once; collect the truth-label columns for PROPERTIES."""
    print(f"[bazaar] Catalog pass: streaming {n_use} rows from {DATASET}")
    ds = load_dataset(DATASET, split="train", streaming=True)
    cols = {p: CATALOG_COLUMNS[p] for p in PROPERTIES}
    bucket: dict[str, list] = {p: [] for p in PROPERTIES}
    for row in tqdm(ds, total=n_use, desc="catalog"):
        for p, c in cols.items():
            bucket[p].append(row[c])
        if len(bucket["redshift"]) >= n_use:
            break
    return {p: np.array(v, dtype=np.float32) for p, v in bucket.items()}


def run_modality(
    telescope: str,
    params: dict[str, np.ndarray],
    basket: list[tuple[str, str]],
    *,
    D: int,
    n_seeds: int,
    n_use: int,
    test_size: int,
    emb_dir,
    whiten_mode: str = "pca_zscore",
) -> list[dict]:
    from bazaar.fit import BazaarFit  # local import: fit.py imports from this module

    if whiten_mode not in WHITEN_MODES:
        raise ValueError(f"whiten_mode must be one of {WHITEN_MODES}, got {whiten_mode!r}")

    rows: list[dict] = []
    print(f"\n[bazaar] === {telescope.upper()} ({whiten_mode}) ===")
    print(f"[bazaar] Loading {len(basket)} embeddings + whiten={whiten_mode}, D={D}...")

    embeddings = load_embeddings(basket, telescope, emb_dir, n_use=n_use)
    fit = BazaarFit.fit(embeddings, basket=basket, D=D, seed=0, whiten_mode=whiten_mode)
    model_names = [f"{f}_{s}" for f, s in basket]
    if whiten_mode == "pca_zscore":
        Z_by_model = {
            name: pca_zscore_transform(embeddings[name], fit.pca[name])
            for name in model_names
        }
        single_tag = f"pca{D}"
    else:
        Z_by_model = {
            name: zscore_transform(embeddings[name], fit.pca[name])
            for name in model_names
        }
        single_tag = "zscore"
    Zs = [Z_by_model[name] for name in model_names]

    # Naive mean and GPA both require identical shapes across models. They
    # only run in pca_zscore mode; zscore mode keeps native widths and is
    # an MCCA-only ablation.
    basket_sources: list[tuple[str, np.ndarray]] = []
    if whiten_mode == "pca_zscore":
        stack = np.stack(Zs, axis=0)
        B_naive = stack.mean(axis=0)
        del stack
        basket_sources.append(("basket_mean", B_naive))

        print(f"[bazaar] {telescope}: running GPA on {len(Zs)} models (D={D})...")
        _, B_proc, gpa_info = generalized_procrustes(
            Zs, max_iter=50, tol=1e-6, verbose=True,
        )
        print(f"[bazaar] {telescope}: GPA converged in {gpa_info['iterations']} iters, "
              f"final mean-Frob² = {gpa_info['loss']:.4e}")
        basket_sources.append(("basket_procrustes_mean", B_proc))
    else:
        widths = [Z.shape[1] for Z in Zs]
        print(f"[bazaar] {telescope}: zscore mode, per-model widths={widths} "
              f"(skipping naive-mean and GPA — heterogeneous shapes)")

    print(f"[bazaar] {telescope}: applying MCCA fit to {len(Zs)} models (D={D})...")
    B_mcca = fit.transform(embeddings)
    basket_sources.append(("basket_mcca_mean", B_mcca))
    print(f"[bazaar] {telescope}: MCCA shared latent shape={B_mcca.shape}")

    for prop in PROPERTIES:
        y = params[prop]
        for seed in range(n_seeds):
            for source, B in basket_sources:
                rows.append(dict(
                    modality=telescope, property=prop, source=source,
                    seed=seed, r2=run_probe(B, y, test_size=test_size, random_state=seed),
                ))
            for name, Z in zip(model_names, Zs):
                rows.append(dict(
                    modality=telescope, property=prop,
                    source=f"single_{name}_{single_tag}",
                    seed=seed,
                    r2=run_probe(Z, y, test_size=test_size, random_state=seed),
                ))

        def _pick(src):
            return [r["r2"] for r in rows
                    if r["modality"] == telescope and r["property"] == prop
                    and r["source"] == src]
        bm = _pick("basket_mcca_mean")
        s_arr = [r["r2"] for r in rows
                 if r["modality"] == telescope and r["property"] == prop
                 and r["source"].startswith("single_")]
        summary = f"  {prop:8s}  mcca={np.mean(bm):.4f}±{np.std(bm):.4f}"
        if whiten_mode == "pca_zscore":
            bn = _pick("basket_mean")
            bp = _pick("basket_procrustes_mean")
            summary = (
                f"  {prop:8s}  naive={np.mean(bn):.4f}±{np.std(bn):.4f}  "
                f"procrustes={np.mean(bp):.4f}±{np.std(bp):.4f}  "
                f"mcca={np.mean(bm):.4f}±{np.std(bm):.4f}"
            )
        print(f"{summary}  best-single={max(s_arr):.4f}  median-single={np.median(s_arr):.4f}")
    return rows
