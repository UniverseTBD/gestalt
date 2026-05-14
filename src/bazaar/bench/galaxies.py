"""End-to-end benchmark sweep on Smith42/galaxies (revision v2.0).

Mirrors `bazaar.bench.gz10.run_gz10` but for `Smith42/galaxies` at
`revision="v2.0"` — which carries both DESI JPEG cutouts and the full
crossmatched metadata in a single row — evaluating the 13 paper-faithful
physical-property targets from Sanjaripour+2026:

  M_g, M_z (absolute magnitudes from photo-z LePhare fit)
  g - r, r - z (DESI apparent colors)
  photo_z, spec_z
  mean sSFR (MPA-JHU total_ssfr_avg, sparsely populated)
  log M_* (LePhare median photo-z mass)
  smooth / disc / artifact fractions (Galaxy Zoo)
  edge-on fraction
  tight-spiral fraction

All targets are continuous regression — the paper's Table 1 reports R² for
both linear and MLP probes; we run the linear half (LinearRegression with
target clipping) for direct comparison to the existing bazaar sweeps.

Pipeline:
  1. Embed all rows through the 22-model basket via `embed_basket`
     (RGB pass-through ingest from `bazaar._ingest.galaxies`).
  2. PCA each model to D, z-score each feature (or skip PCA in zscore mode).
  3. Build basket sources: full-rank whitened MCCA + concat→PCA.
  4. Run `run_probe` for every (target, source, seed).

Output is long-form parquet: one row per (target, source, seed). One run
covers a full split since galaxies is single-modality (legacysurvey).
"""
from __future__ import annotations

import gc

import numpy as np
from sklearn.decomposition import PCA

from bazaar._ingest.galaxies import (
    GALAXIES_DATASET,
    GALAXIES_MODALITY,
    GALAXIES_REVISION,
    GALAXIES_TARGETS,
    galaxies_source,
    stream_image_dr8_ids,
    stream_labels,
)
from bazaar.align import mcca_fit
from bazaar.basket import BASKET, basket_signature
from bazaar.bench.probe import run_probe
from bazaar.embed import embed_basket
from bazaar.whiten import pca_zscore_fit, zscore_fit

WHITEN_MODES = ("pca_zscore", "zscore")

# All 13 targets are continuous regression; ordering pinned for stable output.
PROPERTIES: list[str] = list(GALAXIES_TARGETS.keys())


def catalog_pass(
    split: str = "test",
    max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Stream paper-faithful physical targets in image-stream row order."""
    print(f"[bazaar.galaxies] Catalog pass: streaming labels from "
          f"{GALAXIES_DATASET}@{GALAXIES_REVISION} "
          f"(split={split}, max_samples={max_samples})")
    # Spot-check row-order parity with the imagery stream (catches drift if
    # a future revision reorders rows — same dataset, but pinned belt+braces).
    image_ids = stream_image_dr8_ids(split=split, max_samples=min(max_samples or 8, 8))
    return stream_labels(
        split=split, max_samples=max_samples, image_dr8_ids=image_ids,
    )


def run_galaxies(
    basket: list[tuple[str, str]] = BASKET,
    *,
    D: int = 256,
    n_seeds: int = 5,
    test_size: int = 5_000,
    split: str = "test",
    max_samples: int | None = None,
    cache_dir=None,
    batch_size: int = 64,
    whiten_mode: str = "pca_zscore",
) -> list[dict]:
    """Full embed + align + probe sweep on Smith42/galaxies (v2.0).

    Returns long-form rows: each row has `property`, `source`, `seed`, `r2`.
    """
    if whiten_mode not in WHITEN_MODES:
        raise ValueError(f"whiten_mode must be one of {WHITEN_MODES}, got {whiten_mode!r}")

    sig = basket_signature(basket)
    source = galaxies_source(
        split=split, max_samples=max_samples, basket_signature=sig,
    )
    print(f"[bazaar.galaxies] === galaxies@{GALAXIES_REVISION} "
          f"({whiten_mode}) D={D} split={split} ===")
    print(f"[bazaar.galaxies] Embedding {len(basket)} models via embed_basket...")
    embeddings = embed_basket(
        source, basket=basket, cache_dir=cache_dir, batch_size=batch_size,
    )
    n_rows = next(iter(embeddings.values())).shape[0]
    params = catalog_pass(split=split, max_samples=n_rows)
    first_target = next(iter(params))
    if params[first_target].shape[0] != n_rows:
        raise RuntimeError(
            f"label/embedding row count mismatch: labels="
            f"{params[first_target].shape[0]} vs embeddings={n_rows}. "
            f"Streaming order may have drifted."
        )
    valid_fracs = {p: float(np.isfinite(v).mean()) for p, v in params.items()}
    print(f"[bazaar.galaxies] {n_rows} rows; valid fractions per target:")
    for p, f in valid_fracs.items():
        print(f"  {p:14s} {f:.3f}")

    model_names = [f"{f}_{s}" for f, s in basket]
    # Per-model features for the single-model probes. Independent of the basket
    # source: pca_zscore reduces to D per model, zscore keeps native widths.
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
    print(f"[bazaar.galaxies] full-rank whitened MCCA on {len(model_names)} models "
          f"(per-model native widths={raw_widths}, sum={sum(raw_widths)}, D={D})...")
    Zs_white = [
        pca_zscore_fit(embeddings[name], D=embeddings[name].shape[1])[0]
        for name in model_names
    ]
    _, B_mcca_whitened = mcca_fit(Zs_white, D=D, seed=0)
    basket_sources.append(("basket_mcca_whitened", B_mcca_whitened))
    print(f"[bazaar.galaxies] full-rank whitened MCCA shared latent shape={B_mcca_whitened.shape}")
    del Zs_white
    gc.collect()

    print(f"[bazaar.galaxies] concat→PCA-to-{D} ablation on "
          f"{sum(raw_widths)}-d raw concatenation...")
    C_raw = np.concatenate(
        [embeddings[name] for name in model_names], axis=1,
    ).astype(np.float32)
    B_concat_pca = PCA(
        n_components=D, svd_solver="randomized", random_state=0,
    ).fit_transform(C_raw).astype(np.float32)
    del C_raw
    gc.collect()
    basket_sources.append(("basket_concat_pca", B_concat_pca))

    rows: list[dict] = []
    for prop in PROPERTIES:
        y = params[prop]
        n_valid = int(np.isfinite(y).sum())
        if n_valid < test_size + 100:
            # `train_test_split` will trip if the valid pool is smaller than
            # `test_size`; shrink the holdout proportionally for sparse targets.
            eff_test = max(50, n_valid // 5)
            print(f"[bazaar.galaxies]   {prop}: only {n_valid} valid rows; "
                  f"shrinking test_size {test_size}→{eff_test}")
        else:
            eff_test = test_size
        for seed in range(n_seeds):
            for source_name, B in basket_sources:
                rows.append(dict(
                    modality=GALAXIES_MODALITY,
                    property=prop,
                    kind="regression",
                    source=source_name,
                    seed=seed,
                    r2=run_probe(B, y, test_size=eff_test, random_state=seed),
                    n_valid=n_valid,
                ))
            for name, Z in zip(model_names, Zs):
                rows.append(dict(
                    modality=GALAXIES_MODALITY,
                    property=prop,
                    kind="regression",
                    source=f"single_{name}_{single_tag}",
                    seed=seed,
                    r2=run_probe(Z, y, test_size=eff_test, random_state=seed),
                    n_valid=n_valid,
                ))

        def _pick(src):
            return [r["r2"] for r in rows
                    if r["property"] == prop and r["source"] == src]

        parts = [f"  {prop:13s}"]
        for src in ("basket_mcca_whitened", "basket_concat_pca"):
            vals = _pick(src)
            tag = src.replace("basket_", "")
            parts.append(f"{tag}={np.mean(vals):.4f}±{np.std(vals):.4f}")
        s_arr = [r["r2"] for r in rows
                 if r["property"] == prop and r["source"].startswith("single_")]
        if s_arr:
            parts.append(f"best-single={max(s_arr):.4f}")
            parts.append(f"median-single={np.median(s_arr):.4f}")
        print("  [r2 ] " + "  ".join(parts))
    return rows


__all__ = [
    "GALAXIES_DATASET",
    "GALAXIES_MODALITY",
    "GALAXIES_REVISION",
    "PROPERTIES",
    "catalog_pass",
    "run_galaxies",
]
