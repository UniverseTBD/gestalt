"""End-to-end benchmark sweep: PCA → align → average → probe.

For each modality (telescope):

  1. Load cached per-model embeddings (`bazaar.basket.ensure_embeddings_downloaded`).
  2. PCA each model to D components, then z-score each feature
     (`bazaar.whiten.pca_zscore_*`).
  3. Compute three basket sources:
     - naive mean (no alignment)
     - Procrustes / GPA-aligned mean
     - MCCA shared latent
  4. Run the linear probe on each basket source and each single-model
     PCA feature for `n_seeds` random train/test splits.

Output is long-form: one row per (modality, property, seed, source). This
is the benchmark harness that produced the published `bazaar` numbers; the
installable surface no longer touches GPA, naive-mean, or the probe.
"""
from __future__ import annotations

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from bazaar.align import generalized_procrustes
from bazaar.basket import DATASET, load_embeddings
from bazaar.bench.probe import run_probe
from bazaar.whiten import pca_zscore_transform, zscore_transform

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
