"""End-to-end benchmark sweep on Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2.

For each modality (telescope):

  1. Load cached per-model embeddings (`bazaar.basket.ensure_embeddings_downloaded`).
  2. PCA each model to D components, then z-score each feature
     (`bazaar.whiten.pca_zscore_*`).
  3. Compute two basket sources:
     - full-rank whitened MCCA (per-view PCA to native rank d_m + z-score,
       i.e. full-rank within-view whitening with no dim reduction)
     - concat→PCA-to-D ablation (no per-model whitening; single global SVD
       on the raw 22-model concatenation)
  4. Run the linear probe on each basket source and each single-model
     PCA feature for `n_seeds` random train/test splits.

Output is long-form: one row per (modality, property, seed, source).
"""
from __future__ import annotations

import gc

import numpy as np
from datasets import load_dataset
from sklearn.decomposition import PCA
from tqdm import tqdm

from bazaar.align import mcca_fit
from bazaar.basket import DATASET, load_embeddings
from bazaar.bench.probe import run_probe
from bazaar.whiten import pca_zscore_fit, zscore_fit

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


def run_cosmosweb(
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
    if whiten_mode not in WHITEN_MODES:
        raise ValueError(f"whiten_mode must be one of {WHITEN_MODES}, got {whiten_mode!r}")

    rows: list[dict] = []
    print(f"\n[bazaar] === {telescope.upper()} ({whiten_mode}) ===")
    print(f"[bazaar] Loading {len(basket)} embeddings + whiten={whiten_mode}, D={D}...")

    embeddings = load_embeddings(basket, telescope, emb_dir, n_use=n_use)
    model_names = [f"{f}_{s}" for f, s in basket]
    # Per-model features for the single-model probes. Independent of the basket
    # source: pca_zscore reduces to D per model (matching upstream `pu`), zscore
    # keeps native widths as a PCA ablation.
    if whiten_mode == "pca_zscore":
        Z_by_model = {
            name: pca_zscore_fit(embeddings[name], D=D, seed=0)[0]
            for name in model_names
        }
        single_tag = f"pca{D}"
    else:
        Z_by_model = {
            name: zscore_fit(embeddings[name])[0]
            for name in model_names
        }
        single_tag = "zscore"
    Zs = [Z_by_model[name] for name in model_names]

    basket_sources: list[tuple[str, np.ndarray]] = []

    raw_widths = [embeddings[name].shape[1] for name in model_names]

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
        bw = _pick("basket_mcca_whitened")
        bc = _pick("basket_concat_pca")
        s_arr = [r["r2"] for r in rows
                 if r["modality"] == telescope and r["property"] == prop
                 and r["source"].startswith("single_")]
        parts = [f"  {prop:8s}"]
        parts.append(f"mcca_whitened={np.mean(bw):.4f}±{np.std(bw):.4f}")
        parts.append(f"concat_pca={np.mean(bc):.4f}±{np.std(bc):.4f}")
        parts.append(f"best-single={max(s_arr):.4f}")
        parts.append(f"median-single={np.median(s_arr):.4f}")
        print("  ".join(parts))
    return rows
