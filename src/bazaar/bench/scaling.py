"""Basket pruning curves — `bazaar bench scaling`.

Sweeps R² vs basket size k under two subset-selection policies:

  * `random`         — k models drawn uniformly without replacement.
  * `one_per_family` — k of the 8 model families picked at random,
                       one model from each. Tests whether architectural
                       diversity dominates over basket size.

For each modality we

  1. load all 22 cached per-model embeddings once,
  2. pre-compute the full-rank per-model whitening once,
  3. for each k × subset_kind × subset_id:
       * MCCA on the whitened subset → `basket_mcca_whitened`
         (same fusion as `bazaar fit`),
       * linear probe on each property × seed.

Output is long-form parquet sharing the cosmos schema, with extra columns
`k`, `subset_kind`, `subset_id`, `subset_members` for the curve plot.
"""
from __future__ import annotations

import gc
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from bazaar.align import mcca_fit
from bazaar.basket import load_embeddings
from bazaar.bench.cosmosweb import PROPERTIES
from bazaar.bench.linear_probe import run_probe
from bazaar.whiten import pca_zscore_fit

Basket = list[tuple[str, str]]
SubsetIter = Iterator[tuple[int, Basket]]


# ---------------------------------------------------------------------------
# Subset iterators
# ---------------------------------------------------------------------------

def iter_random_subsets(
    basket: Basket, k: int, n_replicates: int, *, seed: int,
) -> SubsetIter:
    """Yield `n_replicates` random k-subsets of `basket` without replacement.

    Indices within each subset are sorted so the canonical member order is
    stable across runs.
    """
    n = len(basket)
    if k <= 0 or k > n:
        return
    rng = np.random.default_rng(seed)
    for subset_id in range(n_replicates):
        idx = np.sort(rng.choice(n, size=k, replace=False))
        yield subset_id, [basket[i] for i in idx]


def iter_one_per_family_subsets(
    basket: Basket, k: int, n_replicates: int, *, seed: int,
) -> SubsetIter:
    """Yield `n_replicates` subsets that pick one model from each of k families.

    The k families are chosen uniformly from the families present in `basket`;
    inside each chosen family one (family, size) tuple is picked uniformly.
    Skipped (no yields) when `k > n_families`.
    """
    families = sorted({fam for fam, _ in basket})
    if k <= 0 or k > len(families):
        return
    by_family: dict[str, Basket] = {
        fam: [(f, s) for f, s in basket if f == fam] for fam in families
    }
    rng = np.random.default_rng(seed)
    for subset_id in range(n_replicates):
        chosen = np.sort(rng.choice(len(families), size=k, replace=False))
        members: Basket = []
        for fi in chosen:
            fam = families[int(fi)]
            sizes = by_family[fam]
            members.append(sizes[int(rng.integers(0, len(sizes)))])
        yield subset_id, members


def iter_full_basket(basket: Basket) -> SubsetIter:
    """Single yield of the full basket. Used at k == len(basket)."""
    yield 0, list(basket)


# ---------------------------------------------------------------------------
# Per-modality whitening cache + per-subset basket sources
# ---------------------------------------------------------------------------

def precompute_full_whitened(
    embeddings: dict[str, np.ndarray], model_names: list[str],
) -> dict[str, np.ndarray]:
    """Full-rank per-model PCA + z-score, cached by model name.

    Matches the per-view whitening used in `bench._runner.build_basket_sources`
    so MCCA output is identical to the standard sweep when the subset is the
    full basket.
    """
    return {
        name: pca_zscore_fit(embeddings[name], D=embeddings[name].shape[1])[0]
        for name in model_names
    }


def build_basket_sources_subset(
    Zs_white_by_name: dict[str, np.ndarray],
    subset_names: list[str],
    D: int,
    *,
    log_prefix: str = "[bazaar.scaling]",
) -> list[tuple[str, np.ndarray]]:
    """Build the MCCA basket source for one subset.

    Consumes the precomputed `Zs_white_by_name` so the expensive per-model
    whitening is paid once per modality rather than once per subset. Output
    matches `bench._runner.build_basket_sources["basket_mcca_whitened"]`,
    which is the fusion used by `bazaar fit`.
    """
    Zs_subset = [Zs_white_by_name[n] for n in subset_names]
    _, B_mcca = mcca_fit(Zs_subset, D=D, seed=0)
    print(f"{log_prefix} MCCA on {len(subset_names)} views → {B_mcca.shape}")
    return [("basket_mcca_whitened", B_mcca)]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _subsets_for_k(
    basket: Basket, k: int, *,
    n_random_per_k: int, n_one_per_family: int, subset_seed: int,
) -> list[tuple[str, SubsetIter]]:
    """Build the (subset_kind, iterator) list to run at a given k.

    At k == len(basket) we emit a single `full` subset and skip the random /
    family draws (they'd all be identical to the full basket).
    """
    if k == len(basket):
        return [("full", iter_full_basket(basket))]
    n_families = len({f for f, _ in basket})
    out: list[tuple[str, SubsetIter]] = [
        ("random", iter_random_subsets(
            basket, k, n_random_per_k, seed=subset_seed + 1000 * k,
        )),
    ]
    if k <= n_families:
        out.append(("one_per_family", iter_one_per_family_subsets(
            basket, k, n_one_per_family, seed=subset_seed + 1000 * k + 1,
        )))
    return out


def run_scaling_cosmos(
    telescope: str,
    params: dict[str, np.ndarray],
    basket: Basket,
    *,
    D: int,
    ks: list[int],
    n_random_per_k: int,
    n_one_per_family: int,
    n_seeds: int,
    n_use: int,
    test_size: int,
    emb_dir: Path,
    whiten_mode: str = "pca_zscore",
    subset_seed: int = 0,
) -> list[dict]:
    """Scaling sweep on one COSMOS-Web telescope.

    Returns long-form rows: each row is one (k, subset_kind, subset_id,
    property, source, seed) probe. Single-model baselines are not produced
    here; pair this parquet with a `bazaar bench cosmos` run if you want
    them as reference lines.
    """
    print(f"\n[bazaar.scaling] === {telescope.upper()} ({whiten_mode}) D={D} ===")
    embeddings = load_embeddings(basket, telescope, emb_dir, n_use=n_use)
    model_names = [f"{f}_{s}" for f, s in basket]

    rows: list[dict] = []
    eff_test_by_prop: dict[str, int] = {}
    for prop in PROPERTIES:
        y = params[prop]
        n_valid = int(np.isfinite(y).sum())
        if n_valid < test_size + 100:
            eff_test = max(50, n_valid // 5)
            print(f"[bazaar.scaling]   {prop}: only {n_valid} valid rows; "
                  f"shrinking test_size {test_size}→{eff_test}")
        else:
            eff_test = test_size
        eff_test_by_prop[prop] = eff_test

    print(f"[bazaar.scaling] Pre-whitening {len(model_names)} models (full-rank)...")
    Zs_white_by_name = precompute_full_whitened(embeddings, model_names)

    for k in ks:
        iters = _subsets_for_k(
            basket, k,
            n_random_per_k=n_random_per_k,
            n_one_per_family=n_one_per_family,
            subset_seed=subset_seed,
        )
        for kind_label, it in iters:
            for subset_id, members in it:
                subset_names = [f"{f}_{s}" for f, s in members]
                tag = f"{telescope} k={k:>2d} {kind_label:<14s} #{subset_id}"
                print(f"[bazaar.scaling] {tag}: {subset_names}")
                basket_sources = build_basket_sources_subset(
                    Zs_white_by_name, subset_names, D=D,
                    log_prefix=f"[bazaar.scaling]   {tag}",
                )
                members_json = json.dumps(subset_names)
                for prop in PROPERTIES:
                    y = params[prop]
                    n_valid = int(np.isfinite(y).sum())
                    eff_test = eff_test_by_prop[prop]
                    for seed in range(n_seeds):
                        for source_name, B in basket_sources:
                            rows.append(dict(
                                modality=telescope, property=prop,
                                kind="regression", source=source_name,
                                seed=seed,
                                r2=run_probe(
                                    B, y, test_size=eff_test,
                                    random_state=seed,
                                ),
                                acc=np.nan, f1=np.nan, n_valid=n_valid,
                                k=k, subset_kind=kind_label,
                                subset_id=subset_id,
                                subset_members=members_json,
                            ))
                _print_subset_summary(rows, telescope, k, kind_label, subset_id)
                del basket_sources
                gc.collect()
    return rows


def _print_subset_summary(
    rows: list[dict], telescope: str, k: int, kind_label: str, subset_id: int,
) -> None:
    """Per-subset summary line: mean R² across properties × seeds, per source."""
    def _pick(src: str) -> list[float]:
        return [
            r["r2"] for r in rows
            if r["modality"] == telescope and r.get("k") == k
            and r.get("subset_kind") == kind_label
            and r.get("subset_id") == subset_id
            and r["source"] == src
        ]
    parts = [f"  → {kind_label:<14s} #{subset_id}"]
    vals = _pick("basket_mcca_whitened")
    if vals:
        parts.append(f"mcca_whitened={np.mean(vals):.4f}±{np.std(vals):.4f}")
    print("  ".join(parts))


__all__ = [
    "Basket",
    "build_basket_sources_subset",
    "iter_full_basket",
    "iter_one_per_family_subsets",
    "iter_random_subsets",
    "precompute_full_whitened",
    "run_scaling_cosmos",
]
