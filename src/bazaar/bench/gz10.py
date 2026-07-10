"""End-to-end benchmark sweep on Galaxy Zoo 10.

Mirrors `bazaar.bench.cosmosweb.run_cosmosweb` but for `UniverseTBD/mmu_gz10`:

  1. Embed all rows through the 22-model basket via `bazaar.embed.embed_basket`
     (using the RGB pass-through ingest in `bazaar._ingest.gz10`).
  2. PCA each model to D, z-score each feature (or skip PCA in `zscore` mode).
  3. Compute two basket sources:
     - full-rank whitened MCCA (per-view PCA to native rank + zscore, then
       MCCA→D)
     - concat→PCA-to-D ablation
  4. Two probes per source:
     - classification on `gz10_label`     → (accuracy, macro-F1)
     - regression    on `redshift`        → R²
     plus the same probes on each single-model PCA feature.

Output is long-form: one row per (property, source, seed, metric). One run
covers the whole gz10 train split since there's only a single modality.
"""
from __future__ import annotations

import numpy as np

from bazaar._ingest.gz10 import GZ10_DATASET, GZ10_MODALITY, gz10_source, stream_labels
from bazaar.basket import BASKET, basket_signature
from bazaar.bench._runner import build_basket_sources, whiten_per_model
from bazaar.bench.linear_probe import run_classification_probe, run_probe
from bazaar.embed import embed_basket

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
    """Run the appropriate probe; return r2/acc/f1 with NaN where inapplicable."""
    if kind == "classification":
        acc, f1 = run_classification_probe(
            B, y, test_size=test_size, random_state=seed,
        )
        return {"r2": np.nan, "acc": acc, "f1": f1}
    elif kind == "regression":
        r2 = run_probe(B, y, test_size=test_size, random_state=seed)
        return {"r2": r2, "acc": np.nan, "f1": np.nan}
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

    Returns long-form rows with the unified schema (modality, property, kind,
    source, seed, r2, acc, f1, n_valid). Metrics inapplicable to a given probe
    kind are NaN so callers can pivot without conditional logic.
    """
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

    model_names = [f"{f}_{s}" for f, s in basket]
    Zs, single_tag = whiten_per_model(embeddings, model_names, D=D, whiten_mode=whiten_mode)
    basket_sources = build_basket_sources(
        embeddings, model_names, D=D, log_prefix="[bazaar.gz10]",
    )

    rows: list[dict] = []
    for prop, kind in PROPERTIES:
        y = params[prop]
        n_valid = int(np.isfinite(y).sum()) if kind == "regression" else int(y.shape[0])
        for seed in range(n_seeds):
            for source_name, B in basket_sources:
                metrics = _probe_one(B, y, kind, test_size=test_size, seed=seed)
                rows.append(dict(
                    modality=GZ10_MODALITY, property=prop, kind=kind,
                    source=source_name, seed=seed, n_valid=n_valid, **metrics,
                ))
            for name, Z in zip(model_names, Zs):
                metrics = _probe_one(Z, y, kind, test_size=test_size, seed=seed)
                rows.append(dict(
                    modality=GZ10_MODALITY, property=prop, kind=kind,
                    source=f"single_{name}_{single_tag}", seed=seed,
                    n_valid=n_valid, **metrics,
                ))

        # Per-property summary print.
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
        for src in ("basket_mcca_whitened", "basket_concat_pca"):
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
            for src in ("basket_mcca_whitened", "basket_concat_pca"):
                vals = _pick(src, "f1")
                tag = src.replace("basket_", "")
                f1_parts.append(f"{tag}={np.mean(vals):.4f}±{np.std(vals):.4f}")
            print("  [f1 ] " + "  ".join(f1_parts))
    return rows
