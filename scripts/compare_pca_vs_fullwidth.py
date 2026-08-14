#!/usr/bin/env python3
"""Per-model PCA-1024 vs raw full-width on COSMOS-Web linear-probe R².

PCA-1024 numbers come straight out of `data/cosmos_1024.parquet` (the
existing D=1024 sweep — single_<id>_pca1024 rows, five seeds × 6 cells).
Raw full-width numbers are computed here from cached per-model .npy
embeddings using the *same* probe protocol (`bazaar.bench.linear_probe.run_probe`:
train/test split per seed, 1st/99th target clipping, StandardScaler,
LinearRegression on raw columns).

Writes `data/full_width_vs_pca_summary.txt`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports]
from scipy.stats import ttest_1samp  # pyright: ignore[reportMissingImports]

from bazaar.basket import BASKET, emb_npy_path
from bazaar.bench.linear_probe import run_probe

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
EMB_DIR = REPO / "embeds"
OUT_TXT = DATA / "full_width_vs_pca_summary.txt"

D = 1024
N_USE = 45_000
TEST_SIZE = 5_000
N_SEEDS = 5
MODALITIES = ["hsc", "jwst"]
PROPERTIES = ["redshift", "mass", "sSFR"]
PROPERTY_LABEL = {"redshift": "z", "mass": "logM", "sSFR": "sSFR"}


def _full_width_probe(modality: str, model_id: str) -> dict[tuple[str, int], float]:
    """Five-seed raw full-width probe R² for one (modality, model).

    Returns {(property, seed): r2}.
    """
    family, _, size = model_id.rpartition("_")
    path = emb_npy_path(EMB_DIR, modality, family, size)
    if not path.exists():
        raise FileNotFoundError(f"{path} — drop this model from the test")
    E = np.load(path, mmap_mode="r")[:N_USE]
    labels = np.load(DATA / "cosmos_labels.npz")

    rows: dict[tuple[str, int], float] = {}
    for prop in PROPERTIES:
        y = labels[prop].astype(np.float32)
        for seed in range(N_SEEDS):
            r2 = run_probe(np.asarray(E), y, test_size=TEST_SIZE, random_state=seed)
            rows[(prop, seed)] = r2
            print(f"    {modality} {model_id} {prop:8s} seed={seed:2d} r2={r2:.4f}", flush=True)
    return rows


def _pca1024_summary() -> pd.DataFrame:
    """Mean ± std per (modality, model, property) for the PCA-1024 singles."""
    df = pd.read_parquet(DATA / "cosmos_1024.parquet")
    s = df[(df.seed < N_SEEDS) & df.source.str.startswith("single_")].copy()
    s["model"] = s.source.str.removeprefix("single_").str.removesuffix("_pca1024")
    out = s.groupby(["modality", "model", "property"])["r2"].agg(["mean", "std"]).reset_index()
    return out.rename(columns={"mean": "pca_mean", "std": "pca_std"})


def _full_width_summary(raw_rows: list[dict]) -> pd.DataFrame:
    """Mean ± std per (modality, model, property) for the raw full-width probes."""
    df = pd.DataFrame(raw_rows)
    out = df.groupby(["modality", "model", "property"])["r2"].agg(["mean", "std"]).reset_index()
    return out.rename(columns={"mean": "full_mean", "std": "full_std"})


def _tost_pvalue(deltas: pd.Series, margin: float = 0.01):
    lower = ttest_1samp(deltas, -margin, alternative="greater")[1]
    upper = ttest_1samp(deltas, margin, alternative="less")[1]
    return max(lower, upper)


def main() -> None:
    model_ids = [f"{f}_{s}" for f, s in BASKET]
    print(
        f"Testing {len(model_ids)} models × {len(MODALITIES)} modalities × {N_SEEDS} seeds × {len(PROPERTIES)} properties"
    )

    cache = DATA / "full_width_per_seed.parquet"
    if cache.exists():
        print(f"Using cached per-seed results from {cache}")
        full_df = pd.read_parquet(cache)
    else:
        raw_rows: list[dict] = []
        for modality in MODALITIES:
            print(f"=== {modality} ===")
            for model in model_ids:
                print(f"  {model}")
                try:
                    scores = _full_width_probe(modality, model)
                except FileNotFoundError as exc:
                    print(f"  SKIP {model}: {exc}")
                    continue
                for (prop, seed), r2 in scores.items():
                    raw_rows.append(
                        {
                            "modality": modality,
                            "model": model,
                            "property": prop,
                            "seed": seed,
                            "r2": r2,
                        }
                    )
        full_df = pd.DataFrame(raw_rows)
        full_df.to_parquet(cache, index=False)
        print(f"Wrote per-seed cache {cache}")

    full_df = full_df[full_df.seed < N_SEEDS].copy()
    pca = _pca1024_summary()
    full = _full_width_summary(full_df.to_dict(orient="records"))

    joined = pca.merge(full, on=["modality", "model", "property"], how="inner")
    joined["delta"] = joined["pca_mean"] - joined["full_mean"]
    joined = joined.sort_values(
        ["modality", "property", "pca_mean"], ascending=[True, True, False]
    ).reset_index(drop=True)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", 200)

    print()
    print(f"=== Per-model, per-cell summary (R², mean over {N_SEEDS} seeds) ===")
    print(joined.to_string(index=False))

    print()
    print(f"=== Overall summary (22 models × 6 cells = {len(joined)} rows) ===")
    print(f"  PCA-1024 wins  : {(joined.delta > 0).sum()} cells")
    print(f"  Full-width wins: {(joined.delta < 0).sum()} cells")
    print(f"  Ties           : {(joined.delta == 0).sum()}")
    print(f"  Mean delta     : {joined.delta.mean():+.4f}")
    print(f"  Median delta   : {joined.delta.median():+.4f}")
    print(f"  Max PCA gain   : {joined.delta.max():+.4f}")
    print(f"  Max full gain  : {joined.delta.min():+.4f}")

    wide_models = {
        f"{family}_{size}"
        for family, size in BASKET
        if np.load(emb_npy_path(EMB_DIR, "hsc", family, size), mmap_mode="r").shape[1] > D
    }
    all_tost = _tost_pvalue(joined.delta)
    wide_tost = _tost_pvalue(joined[joined.model.isin(wide_models)].delta)
    print(f"  TOST p (all 132 cell means): {all_tost:.3g}")
    print(f"  TOST p (60 native-width >1024 cell means): {wide_tost:.3g}")

    OUT_TXT.write_text(
        joined.to_string(index=False) + "\n\nSummary: "
        f"mean={joined.delta.mean():+.4f}, "
        f"median={joined.delta.median():+.4f}, "
        f"pca_wins={(joined.delta > 0).sum()}, "
        f"full_wins={(joined.delta < 0).sum()}, "
        f"tost_all={all_tost:.3g}, "
        f"tost_wide={wide_tost:.3g}\n"
    )
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
