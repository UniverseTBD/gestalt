#!/usr/bin/env python3
"""Basket-vs-singles strip plot for the canonical COSMOS-Web D=1024 sweep.

Per (modality, property) cell: every single model's probe R² (mean ± SE over
five seeds, sorted by mean) as a strip of points, with horizontal bands for the
two basket sources (MCCA and concat→PCA), a star on the best single and a
dashed line at the median single. Companion stats file tabulates per-cell
means, basket ranks among the 23 sources, and paired two-sided Wilcoxon
p-values (basket vs best single, per-seed pairs).

Resurrected from `bench/plotting.py` (deleted at 415d741, last at 8e9fe4d)
and restyled to the `scripts/plot_*.py` figure family. Reads
`data/results_native_cosmos.parquet`; writes
`figs/basket_vs_singles_summary_cosmos1024.pdf` (paper Fig 1) and
`figs/basket_vs_singles_stats_cosmos1024.txt`.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGS = REPO / "assets" / "plots"

MODALITIES = ["hsc", "jwst"]
PROPERTIES = ["redshift", "mass", "sSFR"]
MODALITY_LABEL = {"hsc": "HSC", "jwst": "JWST"}
PROPERTY_LABEL = {
    "redshift": r"$z_{\rm phot}$",
    "mass": r"$\log M_\star$",
    "sSFR": r"sSFR",
}

BASKET_STYLE = {
    "basket_mcca_whitened": {"label": "Gestalt", "color": "#ff7f0e"},
    "basket_concat_pca": {"label": r"Basket (concat$\to$PCA)", "color": "#17becf"},
}


def _inward_ticks(ax) -> None:
    for which in ("major", "minor"):
        ax.tick_params(axis="x", direction="in", which=which)
        ax.tick_params(axis="y", direction="in", which=which)


def _dedup_legend(axes) -> tuple[list, list]:
    seen: dict[str, object] = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen[lab] = h
    return list(seen.values()), list(seen.keys())


def _short_name(source: str) -> str:
    return re.sub(r"_(?:pca\d+|native)$", "", source.removeprefix("single_"))


def cell_pivot(
    df: pd.DataFrame, modality: str, prop: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-seed R² arrays for one (modality, property) cell: (singles, baskets)."""
    sub = df[(df["modality"] == modality) & (df["property"] == prop)]
    singles: dict[str, np.ndarray] = {}
    baskets: dict[str, np.ndarray] = {}
    for src, grp in sub.groupby("source"):
        arr = grp.sort_values("seed")["r2"].to_numpy()
        if src.startswith("single_"):
            singles[_short_name(src)] = arr
        elif src in BASKET_STYLE:
            baskets[src] = arr
    return singles, baskets


def plot_cell(ax, singles: dict, baskets: dict) -> None:
    names = sorted(singles, key=lambda n: singles[n].mean())
    means = np.array([singles[n].mean() for n in names])
    ses = np.array([singles[n].std(ddof=1) / np.sqrt(len(singles[n])) for n in names])

    x = np.arange(len(names))
    ax.errorbar(
        x,
        means,
        yerr=ses,
        fmt="o",
        color="#1f77b4",
        ms=3,
        capsize=1.5,
        capthick=0.6,
        elinewidth=0.6,
        label="single model (native width)",
    )

    for src, arr in baskets.items():
        style = BASKET_STYLE[src]
        bm = arr.mean()
        bse = arr.std(ddof=1) / np.sqrt(len(arr))
        ax.axhspan(bm - bse, bm + bse, color=style["color"], alpha=0.18, zorder=0)
        ax.axhline(bm, color=style["color"], lw=1.4, label=style["label"])

    best = int(np.argmax(means))
    ax.scatter(
        [best], [means[best]], color="#2ca02c", s=55, marker="*", zorder=5, label="best single"
    )
    ax.axhline(float(np.median(means)), color="#7f7f7f", lw=0.9, ls="--", label="median single")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=80, ha="right", fontsize=4.5)
    ax.grid(True, alpha=0.25)
    _inward_ticks(ax)


def stats_lines(df: pd.DataFrame) -> list[str]:
    lines = [
        f"{'modality':<6}{'property':<10}"
        f"{'mcca_whitened':>20}{'concat_pca':>20}"
        f"{'best_single':>26}{'median_single':>15}"
        f"{'rank_mcca':>10}{'rank_concat':>12}"
        f"{'p_mcca':>10}{'p_concat':>10}"
    ]
    for modality in MODALITIES:
        for prop in PROPERTIES:
            singles, baskets = cell_pivot(df, modality, prop)
            means = {n: a.mean() for n, a in singles.items()}
            best_name = max(means, key=lambda name: means[name])
            best_arr = singles[best_name]
            median_single = float(np.median(list(means.values())))

            cols: dict[str, str] = {}
            ranks: dict[str, int] = {}
            ps: dict[str, float] = {}
            for src in BASKET_STYLE:
                arr = baskets.get(src)
                if arr is None:
                    cols[src], ranks[src], ps[src] = "—", -1, float("nan")
                    continue
                cols[src] = f"{arr.mean():.3f}±{arr.std(ddof=1):.3f}"
                ranks[src] = sum(1 for v in means.values() if v > arr.mean()) + 1
                _, ps[src] = wilcoxon(arr, best_arr, alternative="two-sided", zero_method="wilcox")

            n_src = len(singles) + 1
            lines.append(
                f"{modality:<6}{prop:<10}"
                f"{cols['basket_mcca_whitened']:>20}"
                f"{cols['basket_concat_pca']:>20}"
                f"{means[best_name]:>14.3f} ({best_name[:8]:<8}) "
                f"{median_single:>13.3f}"
                f"{ranks['basket_mcca_whitened']:>7d}/{n_src}"
                f"{ranks['basket_concat_pca']:>9d}/{n_src}"
                f"{ps['basket_mcca_whitened']:>10.3g}"
                f"{ps['basket_concat_pca']:>10.3g}"
            )
    return lines


def main() -> None:
    df = pd.read_parquet(DATA / "results_native_cosmos.parquet")
    df = df[df["seed"] < 5]
    df = df[df["kind"] == "regression"] if "kind" in df.columns else df
    FIGS.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), sharex=False)
    for i, modality in enumerate(MODALITIES):
        for j, prop in enumerate(PROPERTIES):
            ax = axes[i, j]
            singles, baskets = cell_pivot(df, modality, prop)
            plot_cell(ax, singles, baskets)
            ax.set_title(f"{MODALITY_LABEL[modality]} — {PROPERTY_LABEL[prop]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(r"$R^2$")
    handles, labels = _dedup_legend(axes.ravel())
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    pdf_path = FIGS / "basket_vs_singles_summary_cosmos1024.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")

    lines = stats_lines(df)
    txt_path = FIGS / "basket_vs_singles_stats_cosmos1024.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {txt_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
