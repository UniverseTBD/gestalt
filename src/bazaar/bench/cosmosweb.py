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

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from bazaar.basket import DATASET, load_embeddings
from bazaar.bench._runner import build_basket_sources, whiten_per_model
from bazaar.bench.probe import run_probe

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
    print(f"\n[bazaar] === {telescope.upper()} ({whiten_mode}) ===")
    print(f"[bazaar] Loading {len(basket)} embeddings + whiten={whiten_mode}, D={D}...")

    embeddings = load_embeddings(basket, telescope, emb_dir, n_use=n_use)
    model_names = [f"{f}_{s}" for f, s in basket]
    Zs, single_tag = whiten_per_model(embeddings, model_names, D=D, whiten_mode=whiten_mode)
    basket_sources = build_basket_sources(
        embeddings, model_names, D=D, log_prefix=f"[bazaar] {telescope}:",
    )

    rows: list[dict] = []
    for prop in PROPERTIES:
        y = params[prop]
        n_valid = int(np.isfinite(y).sum())
        for seed in range(n_seeds):
            for source, B in basket_sources:
                rows.append(dict(
                    modality=telescope, property=prop, kind="regression",
                    source=source, seed=seed,
                    r2=run_probe(B, y, test_size=test_size, random_state=seed),
                    acc=np.nan, f1=np.nan, n_valid=n_valid,
                ))
            for name, Z in zip(model_names, Zs):
                rows.append(dict(
                    modality=telescope, property=prop, kind="regression",
                    source=f"single_{name}_{single_tag}", seed=seed,
                    r2=run_probe(Z, y, test_size=test_size, random_state=seed),
                    acc=np.nan, f1=np.nan, n_valid=n_valid,
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
