"""End-to-end benchmark sweep on Galaxy Zoo 10.

Mirrors `bazaar.bench.pipeline.run_modality` but for `UniverseTBD/mmu_gz10`:

  1. Embed all rows through the 22-model basket via `bazaar.embed.embed_basket`
     (using the RGB pass-through ingest in `bazaar._ingest.gz10`).
  2. PCA each model to D, z-score each feature (or skip PCA in `zscore` mode).
  3. Compute the same basket sources as the cosmosweb bench:
     - naive mean (pca_zscore only, homogeneous widths)
     - GPA / Procrustes-aligned mean (ditto)
     - MCCA shared latent
     - mass-balanced MCCA (1/sqrt(d_eff) per view)
     - zscore-only MCCA (no per-model PCA)
     - concat→PCA-to-D ablation
  4. Two probes per source:
     - classification on `gz10_label`     → (accuracy, macro-F1)
     - regression    on `redshift`        → R²
     plus the same probes on each single-model PCA feature.

Output is long-form: one row per (property, source, seed, metric). One run
covers the whole gz10 train split since there's only a single modality.
"""
from __future__ import annotations

import gc

import numpy as np
from sklearn.decomposition import PCA

from bazaar._ingest.gz10 import GZ10_DATASET, GZ10_MODALITY, gz10_source, stream_labels
from bazaar.align import generalized_procrustes, mcca_fit
from bazaar.basket import BASKET, basket_signature
from bazaar.bench.probe import run_classification_probe, run_probe
from bazaar.embed import embed_basket
from bazaar.fit import BazaarFit
from bazaar.whiten import pca_zscore_transform, zscore_fit, zscore_transform

WHITEN_MODES = ("pca_zscore", "zscore")

# (property, kind). `kind` picks the probe.
PROPERTIES: list[tuple[str, str]] = [
    ("gz10_label", "classification"),
    ("redshift",   "regression"),
]


def catalog_pass(
    split: str = "train",
    max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Stream gz10_label + redshift in the same row order as the embedding pass."""
    print(f"[bazaar.gz10] Catalog pass: streaming labels from {GZ10_DATASET}"
          f" (split={split}, max_samples={max_samples})")
    return stream_labels(split=split, max_samples=max_samples)


def _probe_one(B: np.ndarray, y: np.ndarray, kind: str, test_size: int, seed: int) -> dict:
    """Run the appropriate probe; return a flat dict of metrics."""
    if kind == "classification":
        acc, f1 = run_classification_probe(
            B, y, test_size=test_size, random_state=seed,
        )
        return {"acc": acc, "f1": f1}
    elif kind == "regression":
        r2 = run_probe(B, y, test_size=test_size, random_state=seed)
        return {"r2": r2}
    else:
        raise ValueError(f"unknown probe kind {kind!r}")


def run_gz10(
    basket: list[tuple[str, str]] = BASKET,
    *,
    D: int = 256,
    n_seeds: int = 5,
    test_size: int = 2_500,
    split: str = "train",
    max_samples: int | None = None,
    cache_dir=None,
    batch_size: int = 64,
    whiten_mode: str = "pca_zscore",
) -> list[dict]:
    """Full embed + align + probe sweep on gz10.

    Returns long-form rows: each row has `property`, `source`, `seed`, plus
    `acc`/`f1` (classification) or `r2` (regression). Untouched metrics are
    `NaN` so callers can pivot the table without conditional logic.
    """
    if whiten_mode not in WHITEN_MODES:
        raise ValueError(f"whiten_mode must be one of {WHITEN_MODES}, got {whiten_mode!r}")

    sig = basket_signature(basket)
    source = gz10_source(split=split, max_samples=max_samples, basket_signature=sig)
    print(f"[bazaar.gz10] === gz10 ({whiten_mode}) D={D} ===")
    print(f"[bazaar.gz10] Embedding {len(basket)} models via embed_basket...")
    embeddings = embed_basket(
        source, basket=basket, cache_dir=cache_dir, batch_size=batch_size,
    )
    n_rows = next(iter(embeddings.values())).shape[0]
    params = catalog_pass(split=split, max_samples=n_rows)
    if params["gz10_label"].shape[0] != n_rows:
        raise RuntimeError(
            f"label/embedding row count mismatch: labels={params['gz10_label'].shape[0]} "
            f"vs embeddings={n_rows}. Streaming order may have drifted."
        )
    print(f"[bazaar.gz10] {n_rows} rows; "
          f"classes={dict(zip(*np.unique(params['gz10_label'], return_counts=True)))} "
          f"redshift_valid_frac={float(np.isfinite(params['redshift']).mean()):.3f}")

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

    basket_sources: list[tuple[str, np.ndarray]] = []
    widths = [Z.shape[1] for Z in Zs]
    homogeneous = len(set(widths)) == 1
    if whiten_mode == "pca_zscore" and homogeneous:
        stack = np.stack(Zs, axis=0)
        basket_sources.append(("basket_mean", stack.mean(axis=0)))
        del stack
        print(f"[bazaar.gz10] running GPA on {len(Zs)} models (D={D})...")
        _, B_proc, gpa_info = generalized_procrustes(
            Zs, max_iter=50, tol=1e-6, verbose=True,
        )
        print(f"[bazaar.gz10] GPA converged in {gpa_info['iterations']} iters, "
              f"final mean-Frob² = {gpa_info['loss']:.4e}")
        basket_sources.append(("basket_procrustes_mean", B_proc))
    else:
        reason = ("zscore mode (native widths)" if whiten_mode == "zscore"
                  else f"D={D} exceeds some model's PCA rank")
        print(f"[bazaar.gz10] {reason}, per-model widths={widths} "
              f"(skipping naive-mean and GPA — heterogeneous shapes)")

    print(f"[bazaar.gz10] applying MCCA fit to {len(Zs)} models (D={D})...")
    B_mcca = fit.transform(embeddings)
    basket_sources.append(("basket_mcca_mean", B_mcca))
    print(f"[bazaar.gz10] MCCA shared latent shape={B_mcca.shape}")

    alphas = [1.0 / np.sqrt(Z.shape[1]) for Z in Zs]
    Zs_eqmass = [alpha * Z for alpha, Z in zip(alphas, Zs)]
    print(f"[bazaar.gz10] running mass-balanced MCCA (1/sqrt(d_eff)) D={D}...")
    _, B_mcca_eqmass = mcca_fit(Zs_eqmass, D=D, seed=0)
    basket_sources.append(("basket_mcca_eqmass", B_mcca_eqmass))
    del Zs_eqmass
    gc.collect()

    raw_widths = [embeddings[name].shape[1] for name in model_names]
    print(f"[bazaar.gz10] zscore-only MCCA (native widths={raw_widths}, "
          f"sum={sum(raw_widths)}) D={D}...")
    Zs_zscore = [zscore_fit(embeddings[name])[0] for name in model_names]
    _, B_mcca_zscore = mcca_fit(Zs_zscore, D=D, seed=0)
    basket_sources.append(("basket_mcca_zscore", B_mcca_zscore))
    del Zs_zscore
    gc.collect()

    print(f"[bazaar.gz10] concat→PCA-to-{D} ablation on "
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
    for prop, kind in PROPERTIES:
        y = params[prop]
        for seed in range(n_seeds):
            for source_name, B in basket_sources:
                metrics = _probe_one(B, y, kind, test_size=test_size, seed=seed)
                rows.append(dict(
                    modality=GZ10_MODALITY,
                    property=prop,
                    kind=kind,
                    source=source_name,
                    seed=seed,
                    **metrics,
                ))
            for name, Z in zip(model_names, Zs):
                metrics = _probe_one(Z, y, kind, test_size=test_size, seed=seed)
                rows.append(dict(
                    modality=GZ10_MODALITY,
                    property=prop,
                    kind=kind,
                    source=f"single_{name}_{single_tag}",
                    seed=seed,
                    **metrics,
                ))

        # Per-property summary print, mirroring run_modality's format.
        def _pick(src, metric):
            return [r[metric] for r in rows
                    if r["property"] == prop and r["source"] == src
                    and metric in r and r[metric] is not None]

        if kind == "classification":
            metric = "acc"
            label = "acc"
        else:
            metric = "r2"
            label = "r2"

        parts = [f"  {prop:11s} ({kind[:5]:5s})"]
        if whiten_mode == "pca_zscore" and homogeneous:
            bn = _pick("basket_mean", metric)
            bp = _pick("basket_procrustes_mean", metric)
            parts.append(f"naive={np.mean(bn):.4f}±{np.std(bn):.4f}")
            parts.append(f"proc={np.mean(bp):.4f}±{np.std(bp):.4f}")
        for src in ("basket_mcca_mean", "basket_mcca_eqmass",
                    "basket_mcca_zscore", "basket_concat_pca"):
            vals = _pick(src, metric)
            tag = src.replace("basket_", "")
            parts.append(f"{tag}={np.mean(vals):.4f}±{np.std(vals):.4f}")
        s_arr = [r[metric] for r in rows
                 if r["property"] == prop and r["source"].startswith("single_")
                 and metric in r]
        if s_arr:
            parts.append(f"best-single={max(s_arr):.4f}")
            parts.append(f"median-single={np.median(s_arr):.4f}")
        print(f"  [{label}] " + "  ".join(parts))

        if kind == "classification":
            # Also print macro-F1 line for the basket sources.
            f1_parts = [f"  {prop:11s} (f1)  "]
            for src in ("basket_mcca_mean", "basket_mcca_eqmass",
                        "basket_mcca_zscore", "basket_concat_pca"):
                vals = _pick(src, "f1")
                tag = src.replace("basket_", "")
                f1_parts.append(f"{tag}={np.mean(vals):.4f}±{np.std(vals):.4f}")
            print("  [f1 ] " + "  ".join(f1_parts))
    return rows
