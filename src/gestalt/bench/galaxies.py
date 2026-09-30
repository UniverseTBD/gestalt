"""End-to-end benchmark sweep on Smith42/galaxies (revision v2.0).

Mirrors `gestalt.bench.gz10.run_gz10` but for `Smith42/galaxies` at
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
target clipping) for direct comparison to the existing gestalt sweeps.

Pipeline:
  1. Embed all rows through the 22-model basket via `embed_basket`
     (RGB pass-through ingest from `gestalt._ingest.galaxies`).
  2. PCA each model to D, z-score each feature (or skip PCA in zscore mode).
  3. Build basket sources: full-rank whitened MCCA + concat→PCA.
  4. Run `run_probe` for every (target, source, seed).

Output is long-form parquet: one row per (target, source, seed). One run
covers a full split since galaxies is single-modality (legacysurvey).
"""

from __future__ import annotations

import numpy as np

from gestalt._ingest.galaxies import (
    GALAXIES_DATASET,
    GALAXIES_MODALITY,
    GALAXIES_REVISION,
    GALAXIES_TARGETS,
    galaxies_source,
    stream_image_dr8_ids,
    stream_labels,
)
from gestalt.basket import BASKET, basket_signature
from gestalt.bench._runner import build_basket_sources, whiten_per_model
from gestalt.bench.linear_probe import run_probe
from gestalt.embed import embed_basket

# All 13 targets are continuous regression; ordering pinned for stable output.
PROPERTIES: list[str] = list(GALAXIES_TARGETS.keys())


def catalog_pass(
    split: str = "test",
    max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Stream paper-faithful physical targets in image-stream row order."""
    print(
        f"[gestalt.galaxies] Catalog pass: streaming labels from "
        f"{GALAXIES_DATASET}@{GALAXIES_REVISION} "
        f"(split={split}, max_samples={max_samples})"
    )
    # Spot-check row-order parity with the imagery stream (catches drift if
    # a future revision reorders rows — same dataset, but pinned belt+braces).
    image_ids = stream_image_dr8_ids(split=split, max_samples=min(max_samples or 8, 8))
    return stream_labels(
        split=split,
        max_samples=max_samples,
        image_dr8_ids=image_ids,
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

    Returns long-form rows with the unified schema (modality, property, kind,
    source, seed, r2, acc, f1, n_valid). All targets are regression so
    `acc` and `f1` are always NaN.
    """
    sig = basket_signature(basket)
    source = galaxies_source(
        split=split,
        max_samples=max_samples,
        basket_signature=sig,
    )
    print(
        f"[gestalt.galaxies] === galaxies@{GALAXIES_REVISION} "
        f"({whiten_mode}) D={D} split={split} ==="
    )
    print(f"[gestalt.galaxies] Embedding {len(basket)} models via embed_basket...")
    embeddings = embed_basket(
        source,
        basket=basket,
        cache_dir=cache_dir,
        batch_size=batch_size,
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
    print(f"[gestalt.galaxies] {n_rows} rows; valid fractions per target:")
    for p, f in valid_fracs.items():
        print(f"  {p:14s} {f:.3f}")

    model_names = [f"{f}_{s}" for f, s in basket]
    Zs, single_tag = whiten_per_model(embeddings, model_names, D=D, whiten_mode=whiten_mode)
    basket_sources = build_basket_sources(
        embeddings,
        model_names,
        D=D,
        log_prefix="[gestalt.galaxies]",
    )

    rows: list[dict] = []
    for prop in PROPERTIES:
        y = params[prop]
        n_valid = int(np.isfinite(y).sum())
        if n_valid < test_size + 100:
            # `train_test_split` will trip if the valid pool is smaller than
            # `test_size`; shrink the holdout proportionally for sparse targets.
            eff_test = max(50, n_valid // 5)
            print(
                f"[gestalt.galaxies]   {prop}: only {n_valid} valid rows; "
                f"shrinking test_size {test_size}→{eff_test}"
            )
        else:
            eff_test = test_size
        for seed in range(n_seeds):
            for source_name, B in basket_sources:
                rows.append(
                    dict(
                        modality=GALAXIES_MODALITY,
                        property=prop,
                        kind="regression",
                        source=source_name,
                        seed=seed,
                        r2=run_probe(B, y, test_size=eff_test, random_state=seed),
                        acc=np.nan,
                        f1=np.nan,
                        n_valid=n_valid,
                    )
                )
            for name, Z in zip(model_names, Zs):
                rows.append(
                    dict(
                        modality=GALAXIES_MODALITY,
                        property=prop,
                        kind="regression",
                        source=f"single_{name}_{single_tag}",
                        seed=seed,
                        r2=run_probe(Z, y, test_size=eff_test, random_state=seed),
                        acc=np.nan,
                        f1=np.nan,
                        n_valid=n_valid,
                    )
                )

        def _pick(src):
            return [r["r2"] for r in rows if r["property"] == prop and r["source"] == src]

        parts = [f"  {prop:13s}"]
        for src in ("basket_mcca_whitened", "basket_concat_pca"):
            vals = _pick(src)
            tag = src.replace("basket_", "")
            parts.append(f"{tag}={np.mean(vals):.4f}±{np.std(vals):.4f}")
        s_arr = [
            r["r2"] for r in rows if r["property"] == prop and r["source"].startswith("single_")
        ]
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
