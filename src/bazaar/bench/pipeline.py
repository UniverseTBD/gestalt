"""End-to-end benchmark sweep: PCA → align → average → probe.

For each modality (telescope):

  1. Load cached per-model embeddings (`bazaar.basket.ensure_embeddings_downloaded`).
  2. PCA each model to D components, then z-score each feature
     (`bazaar.whiten.pca_zscore_*`).
  3. Compute six basket sources:
     - naive mean (no alignment)
     - Procrustes / GPA-aligned mean
     - MCCA shared latent
     - mass-balanced MCCA (each view scaled by 1/sqrt(d_eff) before the SVD,
       so wide and narrow models contribute equal Frobenius mass — SUM-COR-
       style GCCA)
     - full-rank whitened MCCA (per-view PCA to native rank d_m + z-score,
       i.e. full-rank within-view whitening with no dim reduction; isolates
       the contribution of the D-truncation step from the MCCA SVD itself)
     - concat→PCA-to-D ablation (no per-model whitening; tests whether
       the per-model standardisation step is doing real work versus a
       single global SVD on the raw 22-model concatenation)
  4. Run the linear probe on each basket source and each single-model
     PCA feature for `n_seeds` random train/test splits.

Output is long-form: one row per (modality, property, seed, source). This
is the benchmark harness that produced the published `bazaar` numbers; the
installable surface no longer touches GPA, naive-mean, or the probe.
"""
from __future__ import annotations

import gc

import numpy as np
from datasets import load_dataset
from sklearn.decomposition import PCA
from tqdm import tqdm

from bazaar.align import generalized_procrustes, mcca_fit
from bazaar.basket import DATASET, load_embeddings
from bazaar.bench.probe import run_probe
from bazaar.whiten import pca_zscore_fit, pca_zscore_transform, zscore_transform

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
    from bazaar.fit import BazaarFit  # local import keeps the import dag clean

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

    # Naive mean and GPA both require identical shapes across models. In
    # pca_zscore mode this usually holds, but when D exceeds some model's
    # PCA rank (native d_in or N), pca_zscore_fit caps that model at
    # D_eff = min(D, d_in, N) and the basket becomes heterogeneous. zscore
    # mode is always heterogeneous (per-model widths stay at native d_m).
    basket_sources: list[tuple[str, np.ndarray]] = []
    widths = [Z.shape[1] for Z in Zs]
    homogeneous = len(set(widths)) == 1
    if whiten_mode == "pca_zscore" and homogeneous:
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
        reason = ("zscore mode (native widths)" if whiten_mode == "zscore"
                  else f"D={D} exceeds some model's PCA rank")
        print(f"[bazaar] {telescope}: {reason}, per-model widths={widths} "
              f"(skipping naive-mean and GPA — heterogeneous shapes; "
              f"MCCA and concat→PCA still run at D={D})")

    print(f"[bazaar] {telescope}: applying MCCA fit to {len(Zs)} models (D={D})...")
    B_mcca = fit.transform(embeddings)
    basket_sources.append(("basket_mcca_mean", B_mcca))
    print(f"[bazaar] {telescope}: MCCA shared latent shape={B_mcca.shape}")

    # Mass-balanced MCCA ablation: scale each view by 1/sqrt(d_eff) before the
    # SVD so all views contribute equal Frobenius mass (SUM-COR-style GCCA).
    # Tests whether wide-model dominance in the standard MCCA SVD is costing R².
    alphas = [1.0 / np.sqrt(Z.shape[1]) for Z in Zs]
    Zs_eqmass = [alpha * Z for alpha, Z in zip(alphas, Zs)]
    print(f"[bazaar] {telescope}: running mass-balanced MCCA on {len(Zs)} models "
          f"(D={D}, view weights=1/sqrt(d_eff))...")
    _, B_mcca_eqmass = mcca_fit(Zs_eqmass, D=D, seed=0)
    basket_sources.append(("basket_mcca_eqmass", B_mcca_eqmass))
    print(f"[bazaar] {telescope}: mass-balanced MCCA shared latent shape={B_mcca_eqmass.shape}")
    del Zs_eqmass
    gc.collect()

    raw_widths = [embeddings[name].shape[1] for name in model_names]

    # Ablation: per-view full-rank PCA + z-score (within-view whitening, no dim
    # reduction) → MCCA→D. Isolates the D-truncation step from the MCCA SVD: if
    # this matches `basket_mcca_mean`, the per-model PCA-to-D is just throwing
    # away signal that the MCCA SVD would have recovered anyway.
    print(f"[bazaar] {telescope}: full-rank whitened MCCA on {len(model_names)} models "
          f"(per-model native widths={raw_widths}, sum={sum(raw_widths)}, D={D})...")
    Zs_white = [
        pca_zscore_fit(embeddings[name], D=embeddings[name].shape[1])[0]
        for name in model_names
    ]
    _, B_mcca_whitened = mcca_fit(Zs_white, D=D, seed=0)
    basket_sources.append(("basket_mcca_whitened", B_mcca_whitened))
    print(f"[bazaar] {telescope}: full-rank whitened MCCA shared latent shape={B_mcca_whitened.shape}")
    del Zs_white
    gc.collect()

    # Ablation: concat raw per-model embeddings (no whitening, no alignment),
    # take the top-D principal components of the full concatenation, and probe.
    # If this matches or beats MCCA, the per-model standardisation step isn't
    # earning its keep — the basket-vs-single win is a parameter-count win.
    print(f"[bazaar] {telescope}: concat→PCA-to-{D} ablation on "
          f"{sum(raw_widths)}-d raw concatenation (widths={raw_widths})...")
    C_raw = np.concatenate(
        [embeddings[name] for name in model_names], axis=1,
    ).astype(np.float32)
    B_concat_pca = PCA(
        n_components=D, svd_solver="randomized", random_state=0,
    ).fit_transform(C_raw).astype(np.float32)
    del C_raw
    gc.collect()
    basket_sources.append(("basket_concat_pca", B_concat_pca))
    print(f"[bazaar] {telescope}: concat→PCA shape={B_concat_pca.shape}")

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
        be = _pick("basket_mcca_eqmass")
        bw = _pick("basket_mcca_whitened")
        bc = _pick("basket_concat_pca")
        s_arr = [r["r2"] for r in rows
                 if r["modality"] == telescope and r["property"] == prop
                 and r["source"].startswith("single_")]
        parts = [f"  {prop:8s}"]
        if whiten_mode == "pca_zscore" and homogeneous:
            bn = _pick("basket_mean")
            bp = _pick("basket_procrustes_mean")
            parts.append(f"naive={np.mean(bn):.4f}±{np.std(bn):.4f}")
            parts.append(f"procrustes={np.mean(bp):.4f}±{np.std(bp):.4f}")
        parts.append(f"mcca={np.mean(bm):.4f}±{np.std(bm):.4f}")
        parts.append(f"mcca_eqmass={np.mean(be):.4f}±{np.std(be):.4f}")
        parts.append(f"mcca_whitened={np.mean(bw):.4f}±{np.std(bw):.4f}")
        parts.append(f"concat_pca={np.mean(bc):.4f}±{np.std(bc):.4f}")
        parts.append(f"best-single={max(s_arr):.4f}")
        parts.append(f"median-single={np.median(s_arr):.4f}")
        print("  ".join(parts))
    return rows
