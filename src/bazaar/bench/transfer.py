"""Cross-survey generalization sweep — `bazaar bench transfer`.

Probes whether a `BazaarFit` trained on one corpus generalizes to another.
For each (target T, source S) pair, fit a `BazaarFit` on the first `n_fit`
rows of S's 22-model embeddings, transform T's full embeddings through it,
then run the standard linear probe on T's labels. A `concat→PCA` projector
fit on the same S rows is run alongside as a transferable baseline.

Cells the sweep is designed to fill (one parquet per target):

  * Cross-survey: cosmos-hsc ↔ {gz10, galaxies}   — the §3.2 headline.
  * Cross-dataset within survey: gz10 ↔ galaxies  — control.
  * Cross-modality within survey: cosmos-hsc ↔ cosmos-jwst — sanity.
  * Native (S == T at the same n_fit):           — baseline.

Every source corpus is capped to `n_fit` rows before fitting so corpus
size is not a confound across (45k HSC, ~17k GZ10, 86k galaxies). The
target is always probed at full row count.

Output schema (per target parquet):

    target          ('cosmos-hsc' | 'cosmos-jwst' | 'gz10' | 'galaxies')
    modality        target's modality
    fit_source      one of the four corpus names
    source          ('basket_mcca_whitened' | 'basket_concat_pca')
    property        target's property name
    kind            ('regression' | 'classification')
    seed            (0..n_seeds-1)
    r2              (NaN for classification)
    acc, f1         (NaN for regression)
    n_valid         rows passing isfinite on the label
    D, n_fit        carried through
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from bazaar.basket import BASKET, basket_signature, load_embeddings
from bazaar.bench.probe import run_classification_probe, run_probe
from bazaar.embed import embed_basket
from bazaar.fit import BazaarFit
from bazaar.whiten import pca_zscore_fit, pca_zscore_transform

CORPORA: tuple[str, ...] = ("cosmos-hsc", "cosmos-jwst", "gz10", "galaxies")

# Single-encoder baseline: the largest astroPT checkpoint. Run alongside the
# basket pipelines so a reader can see whether the basket's cross-domain
# plateau beats (or matches) a strong single encoder under the same fit/
# transfer protocol — i.e. isolate whether MCCA's V is the brittle piece or
# whether per-encoder PCA-fit drift is what costs R² off-domain.
SINGLE_MODEL_BASELINE: tuple[str, str] = ("astropt", "850M")

# (corpus name → target modality string in the bench output schema).
CORPUS_MODALITY: dict[str, str] = {
    "cosmos-hsc":  "hsc",
    "cosmos-jwst": "jwst",
    "gz10":        "legacysurvey",
    "galaxies":    "legacysurvey",
}


# ---------------------------------------------------------------------------
# Per-corpus embedding loaders
# ---------------------------------------------------------------------------

def load_corpus_embeddings(
    name: str,
    *,
    emb_dir: Path,
    cache_dir: Path | None = None,
    n_use: int | None = None,
    split: str | None = None,
    max_samples: int | None = None,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Load the full 22-model embedding dict for a named corpus.

    HSC/JWST come from the per-telescope `.npy` cache in `emb_dir`.
    GZ10/galaxies go through `embed_basket`, which hits the per-fingerprint
    cache under `cache_dir` if a previous bench run produced it (otherwise
    embeds, which requires CUDA + the upstream HF dataset to be reachable).
    """
    if name == "cosmos-hsc":
        return load_embeddings(BASKET, "hsc", emb_dir, n_use=n_use or 45_000)
    if name == "cosmos-jwst":
        return load_embeddings(BASKET, "jwst", emb_dir, n_use=n_use or 45_000)
    if name == "gz10":
        from bazaar._ingest.gz10 import gz10_source
        source = gz10_source(
            split=split or "train", max_samples=max_samples,
            basket_signature=basket_signature(BASKET),
        )
        return embed_basket(
            source, basket=BASKET, cache_dir=cache_dir, batch_size=batch_size,
        )
    if name == "galaxies":
        from bazaar._ingest.galaxies import galaxies_source
        source = galaxies_source(
            split=split or "test", max_samples=max_samples,
            basket_signature=basket_signature(BASKET),
        )
        return embed_basket(
            source, basket=BASKET, cache_dir=cache_dir, batch_size=batch_size,
        )
    raise ValueError(f"unknown corpus {name!r}; expected one of {CORPORA}")


def load_corpus_labels(
    name: str,
    *,
    n_rows: int,
    split: str | None = None,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    """Return ({property → labels[N_rows]}, [(property, probe_kind), ...]).

    `probe_kind` is "regression" or "classification" so callers can pick the
    right probe per property.
    """
    if name in ("cosmos-hsc", "cosmos-jwst"):
        from bazaar.bench.cosmosweb import PROPERTIES, catalog_pass
        labels = catalog_pass(n_rows)
        return labels, [(p, "regression") for p in PROPERTIES]
    if name == "gz10":
        from bazaar.bench.gz10 import PROPERTIES, catalog_pass
        labels = catalog_pass(split=split or "train", max_samples=n_rows)
        return labels, [(p, k) for p, k in PROPERTIES]
    if name == "galaxies":
        from bazaar.bench.galaxies import PROPERTIES, catalog_pass
        labels = catalog_pass(split=split or "test", max_samples=n_rows)
        return labels, [(p, "regression") for p in PROPERTIES]
    raise ValueError(f"unknown corpus {name!r}; expected one of {CORPORA}")


# ---------------------------------------------------------------------------
# Fit + transfer primitives
# ---------------------------------------------------------------------------

def _slice_to_n_fit(embeddings: dict[str, np.ndarray], n_fit: int) -> dict[str, np.ndarray]:
    return {k: np.ascontiguousarray(v[:n_fit]) for k, v in embeddings.items()}


def _concat_in_basket_order(
    embeddings: dict[str, np.ndarray], model_names: list[str],
) -> np.ndarray:
    return np.concatenate(
        [embeddings[m] for m in model_names], axis=1,
    ).astype(np.float32, copy=False)


def fit_source(
    name: str,
    src_embeddings: dict[str, np.ndarray],
    *,
    n_fit: int,
    D: int,
    seed: int,
    model_names: list[str],
    single_baseline: tuple[str, str] | None = SINGLE_MODEL_BASELINE,
    log_prefix: str = "[bazaar.transfer]",
) -> tuple[BazaarFit, PCA, dict[str, np.ndarray] | None]:
    """Fit the basket and the single-encoder baseline on the first `n_fit` rows.

    Returns (BazaarFit, concat-PCA projector, single-encoder pca_zscore
    artifacts) — all fit on the same rows. The single-encoder artifacts are
    None when `single_baseline` is None or the model isn't in `src_embeddings`.
    """
    if next(iter(src_embeddings.values())).shape[0] < n_fit:
        raise ValueError(
            f"{name!r} only has {next(iter(src_embeddings.values())).shape[0]} "
            f"rows but n_fit={n_fit}"
        )

    sub = _slice_to_n_fit(src_embeddings, n_fit)
    print(f"{log_prefix} fit BazaarFit on {name!r} ({n_fit} rows, D={D})...")
    bf = BazaarFit.fit(sub, basket=BASKET, D=D, seed=seed)

    print(f"{log_prefix} fit concat→PCA-{D} on {name!r} ({n_fit} rows)...")
    src_concat = _concat_in_basket_order(sub, model_names)
    pca_proj = PCA(
        n_components=D, svd_solver="randomized", random_state=seed,
    ).fit(src_concat)
    del src_concat
    gc.collect()

    single_artifacts: dict[str, np.ndarray] | None = None
    if single_baseline is not None:
        key = f"{single_baseline[0]}_{single_baseline[1]}"
        if key in sub:
            d_in = sub[key].shape[1]
            print(f"{log_prefix} fit single-encoder {key} PCA→{min(D, d_in)} "
                  f"+ z-score on {name!r} ({n_fit} rows)...")
            _, single_artifacts = pca_zscore_fit(sub[key], D=D, seed=seed)
        else:
            print(f"{log_prefix} single-encoder {key} not in source "
                  f"embeddings — skipping baseline.")

    del sub
    gc.collect()
    return bf, pca_proj, single_artifacts


def transfer_to_target(
    target: str,
    target_embeddings: dict[str, np.ndarray],
    target_concat: np.ndarray,
    target_labels: dict[str, np.ndarray],
    target_properties: list[tuple[str, str]],
    *,
    source_name: str,
    bf: BazaarFit,
    pca_proj: PCA,
    D: int,
    n_seeds: int,
    test_size: int,
    n_fit: int,
    single_baseline: tuple[str, str] | None = SINGLE_MODEL_BASELINE,
    single_artifacts: dict[str, np.ndarray] | None = None,
    log_prefix: str = "[bazaar.transfer]",
) -> list[dict]:
    """Project target through the source-trained fits; probe per (property, seed).

    Returns long-form rows tagged with `fit_source=source_name`.
    """
    print(f"{log_prefix} transform target {target!r} via fit_source={source_name!r}...")
    S_mcca = bf.transform(target_embeddings).astype(np.float32, copy=False)
    S_concat = pca_proj.transform(target_concat).astype(np.float32, copy=False)
    reps: list[tuple[str, np.ndarray]] = [
        ("basket_mcca_whitened", S_mcca),
        ("basket_concat_pca",    S_concat),
    ]
    if single_baseline is not None and single_artifacts is not None:
        key = f"{single_baseline[0]}_{single_baseline[1]}"
        if key in target_embeddings:
            S_single = pca_zscore_transform(target_embeddings[key], single_artifacts)
            reps.append((f"single_{key}_pca_zscore", S_single))
        else:
            print(f"{log_prefix} single-encoder {key} not in target embeddings — "
                  f"skipping baseline.")
    representations = tuple(reps)

    modality = CORPUS_MODALITY[target]
    rows: list[dict] = []
    for prop, kind in target_properties:
        y = target_labels[prop]
        if kind == "regression":
            n_valid = int(np.isfinite(y).sum())
        else:
            n_valid = int(y.shape[0])
        for source_label, B in representations:
            for seed in range(n_seeds):
                if kind == "regression":
                    r2 = run_probe(B, y, test_size=test_size, random_state=seed)
                    metrics = {"r2": float(r2), "acc": float("nan"), "f1": float("nan")}
                else:
                    acc, f1 = run_classification_probe(
                        B, y, test_size=test_size, random_state=seed,
                    )
                    metrics = {"r2": float("nan"), "acc": float(acc), "f1": float(f1)}
                rows.append(dict(
                    target=target, modality=modality, fit_source=source_name,
                    source=source_label, property=prop, kind=kind, seed=seed,
                    n_valid=n_valid, D=D, n_fit=n_fit, **metrics,
                ))
    return rows


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_transfer(
    target: str,
    sources: list[str],
    *,
    n_fit: int = 10_000,
    D: int = 1024,
    n_seeds: int = 5,
    test_size: int = 2_500,
    emb_dir: Path | None = None,
    cache_dir: Path | None = None,
    target_split: str | None = None,
    target_max_samples: int | None = None,
    target_n_use: int | None = None,
    source_split: str | None = None,
    source_max_samples: int | None = None,
    source_n_use: int | None = None,
    batch_size: int = 64,
    seed: int = 0,
) -> list[dict]:
    """Cross-survey transfer sweep for one target × many source corpora."""
    if target not in CORPORA:
        raise ValueError(f"unknown target {target!r}; expected one of {CORPORA}")
    unknown = [s for s in sources if s not in CORPORA]
    if unknown:
        raise ValueError(f"unknown source(s) {unknown!r}; expected from {CORPORA}")

    print(f"\n[bazaar.transfer] === target={target!r} sources={sources!r} "
          f"D={D} n_fit={n_fit} n_seeds={n_seeds} ===")

    model_names = [f"{f}_{s}" for f, s in BASKET]

    # ----- Target side: full embeddings + labels (once) ---------------------
    target_emb = load_corpus_embeddings(
        target, emb_dir=emb_dir, cache_dir=cache_dir,
        n_use=target_n_use, split=target_split,
        max_samples=target_max_samples, batch_size=batch_size,
    )
    n_target = next(iter(target_emb.values())).shape[0]
    target_labels, target_properties = load_corpus_labels(
        target, n_rows=n_target, split=target_split,
    )
    first_prop = target_properties[0][0]
    if target_labels[first_prop].shape[0] != n_target:
        raise RuntimeError(
            f"label/embedding row-count mismatch for target {target!r}: "
            f"labels={target_labels[first_prop].shape[0]} vs embeddings={n_target}. "
            f"Streaming order may have drifted."
        )
    print(f"[bazaar.transfer] target {target!r}: N={n_target}, "
          f"properties={[p for p, _ in target_properties]}")

    target_concat = _concat_in_basket_order(target_emb, model_names)
    print(f"[bazaar.transfer] target concat shape={target_concat.shape}")

    # ----- Loop over sources -----------------------------------------------
    rows: list[dict] = []
    for source_name in sources:
        if source_name == target:
            src_emb_full = target_emb  # reuse — no second load
            print(f"\n[bazaar.transfer] --- fit_source={source_name!r} (native, "
                  f"first {n_fit} rows of target) ---")
        else:
            print(f"\n[bazaar.transfer] --- fit_source={source_name!r} ---")
            src_emb_full = load_corpus_embeddings(
                source_name, emb_dir=emb_dir, cache_dir=cache_dir,
                n_use=source_n_use, split=source_split,
                max_samples=source_max_samples, batch_size=batch_size,
            )

        bf, pca_proj, single_artifacts = fit_source(
            source_name, src_emb_full, n_fit=n_fit, D=D, seed=seed,
            model_names=model_names,
        )

        if source_name != target:
            del src_emb_full
            gc.collect()

        rows.extend(transfer_to_target(
            target, target_emb, target_concat, target_labels, target_properties,
            source_name=source_name, bf=bf, pca_proj=pca_proj,
            single_artifacts=single_artifacts,
            D=D, n_seeds=n_seeds, test_size=test_size, n_fit=n_fit,
        ))

        # Per-source summary print.
        single_key = f"{SINGLE_MODEL_BASELINE[0]}_{SINGLE_MODEL_BASELINE[1]}"
        single_label = f"single_{single_key}_pca_zscore"
        for prop, kind in target_properties:
            metric = "r2" if kind == "regression" else "acc"
            def pick(rep: str) -> list[float]:
                return [
                    r[metric] for r in rows
                    if r["fit_source"] == source_name and r["property"] == prop
                    and r["source"] == rep
                ]
            mcca = pick("basket_mcca_whitened")
            concat = pick("basket_concat_pca")
            single = pick(single_label)
            line = (f"  {prop:14s} ({kind[:5]}) "
                    f"mcca={np.mean(mcca):.4f}±{np.std(mcca):.4f}  "
                    f"concat={np.mean(concat):.4f}±{np.std(concat):.4f}")
            if single:
                line += f"  {single_key}={np.mean(single):.4f}±{np.std(single):.4f}"
            print(line)

        del bf, pca_proj, single_artifacts
        gc.collect()

    return rows


__all__ = [
    "CORPORA",
    "CORPUS_MODALITY",
    "SINGLE_MODEL_BASELINE",
    "load_corpus_embeddings",
    "load_corpus_labels",
    "fit_source",
    "transfer_to_target",
    "run_transfer",
]
