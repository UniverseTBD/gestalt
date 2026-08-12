#!/usr/bin/env python3
"""Basket-vs-singles strip plot, pooled across the three COSMOS properties.

Pre-collapse: for each (source, modality, seed) take the mean probe R² over
{redshift, mass, sSFR}. The plotted dots are the mean of those 3 values
(one per seed) with SE over the 10 seeds. Baskets (MCCA stitch, concat→PCA)
are drawn as horizontal bands off the same body of collapsed seed values.

Companion file lists per-model means, ranks, and paired two-sided Wilcoxon
p-values on the collapsed seed vectors.

Reads `data/cosmos_1024.parquet`; writes
`figs/basket_vs_singles_pooled_cosmos1024.pdf` and
`figs/basket_vs_singles_pooled_cosmos1024.txt`.
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
MODALITY_LABEL = {"hsc": "HSC", "jwst": "JWST"}

BASKET_STYLE = {
    "basket_mcca_whitened": {"label": "Basket (MCCA, whitened)", "color": "#ff7f0e"},
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
    return re.sub(r"_pca\d+$", "", source.removeprefix("single_"))


def cell_pivot(
    df: pd.DataFrame, modality: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Collapsed per-seed R² (mean over 3 properties) per source."""
    sub = df[df["modality"] == modality]
    collapsed = sub.groupby(["source", "seed"])["r2"].mean().reset_index(name="r2_mean3")
    singles: dict[str, np.ndarray] = {}
    baskets: dict[str, np.ndarray] = {}
    for src, grp in collapsed.groupby("source"):
        arr = grp.sort_values("seed")["r2_mean3"].to_numpy()
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
        label="single model (PCA)",
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
        f"{'modality':<6}{'mcca_whitened':>20}{'concat_pca':>20}"
        f"{'best_single':>26}{'median_single':>15}"
        f"{'rank_mcca':>10}{'rank_concat':>12}"
        f"{'p_mcca':>10}{'p_concat':>10}"
    ]
    for modality in MODALITIES:
        singles, baskets = cell_pivot(df, modality)
        means = {n: a.mean() for n, a in singles.items()}
        best_name = max(means, key=means.get)
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
            f"{modality:<6}"
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
    df = pd.read_parquet(DATA / "cosmos_1024.parquet")
    df = df[df["kind"] == "regression"] if "kind" in df.columns else df
    FIGS.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.2), sharey=True)
    for i, modality in enumerate(MODALITIES):
        ax = axes[i]
        singles, baskets = cell_pivot(df, modality)
        plot_cell(ax, singles, baskets)
        ax.set_title(
            f"{MODALITY_LABEL[modality]} — mean over "
            r"$\{z_{\rm phot}, \log M_\star, {\rm sSFR}\}$",
            fontsize=9,
        )
        if i == 0:
            ax.set_ylabel(r"$R^2$")
    handles, labels = _dedup_legend(axes.ravel())
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    pdf_path = FIGS / "basket_vs_singles_pooled_cosmos1024.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")

    lines = stats_lines(df)
    txt_path = FIGS / "basket_vs_singles_pooled_cosmos1024.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {txt_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
