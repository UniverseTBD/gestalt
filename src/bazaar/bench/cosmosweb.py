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


# ---------------------------------------------------------------------------
# Wide catalog pass used by `bazaar bench dims`.
# ---------------------------------------------------------------------------
#
# The basic `catalog_pass` only pulls (z, mass, sSFR). For the per-dimension
# covariate decomposition (`bench dims`) we need a much wider set of physical,
# photometric, morphological, systematic, and positional covariates. Grouping
# is exposed alongside the covariate values so the heatmap can render rows in
# their natural blocks.

# (covariate_name, dataset_column) — direct one-to-one pulls.
_WIDE_DIRECT_COLUMNS: dict[str, tuple[str, str]] = {
    # group, column
    "z":              ("physics",     "lephare_photozs"),
    "log_M_star":     ("physics",     "lp_mass"),
    "log_SFR":        ("physics",     "lp_sfr"),
    "log_sSFR":       ("physics",     "lp_ssfr"),
    "log_age":        ("physics",     "lp_age"),
    "mag_u":          ("photometry",  "mag_model_cfht-u"),
    "mag_g":          ("photometry",  "mag_model_hsc-g"),
    "mag_r":          ("photometry",  "mag_model_hsc-r"),
    "mag_i":          ("photometry",  "mag_model_hsc-i"),
    "mag_z":          ("photometry",  "mag_model_hsc-z"),
    "mag_y":          ("photometry",  "mag_model_hsc-y"),
    "mag_F115W":      ("photometry",  "mag_model_f115w"),
    "mag_F150W":      ("photometry",  "mag_model_f150w"),
    "mag_F277W":      ("photometry",  "mag_model_f277w"),
    "mag_F444W":      ("photometry",  "mag_model_f444w"),
    "bulge_radius":   ("morphology",  "bulge_radius"),
    "disk_radius":    ("morphology",  "disk_radius"),
    "specz_conf":     ("systematics", "Confidence_level"),
    "ra":             ("position",    "ra"),
    "dec":            ("position",    "dec"),
}

# (numerator_column, denominator_column) for SNR proxies — one per band.
_WIDE_SNR_BANDS: list[tuple[str, str]] = [
    ("u",     "cfht-u"),
    ("g",     "hsc-g"),
    ("r",     "hsc-r"),
    ("i",     "hsc-i"),
    ("z",     "hsc-z"),
    ("y",     "hsc-y"),
    ("F115W", "f115w"),
    ("F150W", "f150w"),
    ("F277W", "f277w"),
    ("F444W", "f444w"),
]

# (covariate_name, blue_mag, red_mag) — colors are differences of pulled mags.
_WIDE_COLOR_PAIRS: list[tuple[str, str, str]] = [
    ("g_minus_r",        "mag_g",     "mag_r"),
    ("r_minus_i",        "mag_r",     "mag_i"),
    ("i_minus_z",        "mag_i",     "mag_z"),
    ("i_minus_y",        "mag_i",     "mag_y"),
    ("F115W_minus_F150W","mag_F115W", "mag_F150W"),
    ("F150W_minus_F277W","mag_F150W", "mag_F277W"),
    ("F277W_minus_F444W","mag_F277W", "mag_F444W"),
    ("u_minus_F444W",    "mag_u",     "mag_F444W"),
]


def catalog_pass_wide(n_use: int) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Stream the dataset once; collect the wide covariate set for `bench dims`.

    Returns
    -------
    covariates : dict[str, ndarray]  — values, length n_use each.
    groups     : dict[str, str]      — covariate_name → covariate_group label.
    """
    print(f"[bazaar] Wide catalog pass: streaming {n_use} rows from {DATASET}")
    ds = load_dataset(DATASET, split="train", streaming=True)

    # Source columns we need to pull from the row dict.
    raw_cols: set[str] = {col for _, col in _WIDE_DIRECT_COLUMNS.values()}
    for _, band in _WIDE_SNR_BANDS:
        raw_cols.add(f"flux_model_{band}")
        raw_cols.add(f"flux_err-cal_model_{band}")
    raw_cols.update({"lp_zpdf_l68", "lp_zpdf_u68"})

    bucket: dict[str, list] = {c: [] for c in raw_cols}
    for row in tqdm(ds, total=n_use, desc="catalog-wide"):
        for c in raw_cols:
            bucket[c].append(row[c])
        if len(bucket[next(iter(raw_cols))]) >= n_use:
            break

    raw = {c: np.array(v, dtype=np.float32) for c, v in bucket.items()}

    covariates: dict[str, np.ndarray] = {}
    groups: dict[str, str] = {}

    # 1. Direct columns.
    for name, (group, col) in _WIDE_DIRECT_COLUMNS.items():
        covariates[name] = raw[col]
        groups[name] = group

    # 2. SNR per band: flux / flux_err (clipped at finite values).
    for tag, band in _WIDE_SNR_BANDS:
        flux = raw[f"flux_model_{band}"]
        ferr = raw[f"flux_err-cal_model_{band}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(ferr > 0, flux / ferr, np.nan).astype(np.float32)
        covariates[f"snr_{tag}"] = snr
        groups[f"snr_{tag}"] = "systematics"

    # 3. Colors: difference of two pulled magnitudes.
    for name, blue, red in _WIDE_COLOR_PAIRS:
        covariates[name] = (covariates[blue] - covariates[red]).astype(np.float32)
        groups[name] = "color"

    # 4. Morphology derived: log10(bulge / disk) with ε guard.
    eps = np.float32(1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_bd = np.log10(covariates["bulge_radius"] + eps) - \
                 np.log10(covariates["disk_radius"] + eps)
    covariates["log_bulge_over_disk"] = log_bd.astype(np.float32)
    groups["log_bulge_over_disk"] = "morphology"

    # 5. zpdf width = u68 - l68.
    covariates["zpdf_width"] = (raw["lp_zpdf_u68"] - raw["lp_zpdf_l68"]).astype(np.float32)
    groups["zpdf_width"] = "systematics"

    return covariates, groups


__all__ = ["catalog_pass", "catalog_pass_wide", "run_cosmosweb", "PROPERTIES", "CATALOG_COLUMNS"]


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
