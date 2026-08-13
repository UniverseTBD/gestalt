#!/usr/bin/env python3
"""Basket-vs-singles strip plot, pooled across the three COSMOS properties.

Pre-collapse: for each (source, modality, seed) take the mean probe R² over
{redshift, mass, sSFR}. The plotted dots are the mean of those 3 values
(one per seed) with SE over five seeds. Gestalt and concat→PCA
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
import pandas as pd  # pyright: ignore[reportMissingImports]
from scipy.stats import wilcoxon

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "DejaVu Sans"],
    }
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGS = REPO / "assets" / "plots"

MODALITIES = ["hsc", "jwst"]
MODALITY_LABEL = {"hsc": "HSC", "jwst": "JWST"}

BASKET_STYLE = {
    "basket_mcca_whitened": {"label": "Gestalt", "color": "#ff7f0e"},
    "basket_concat_pca": {"label": r"Basket (concat$\to$PCA)", "color": "#17becf"},
}
MODEL_LABELS = {
    "astropt_015M": "AstroPT 15M",
    "astropt_095M": "AstroPT 95M",
    "astropt_850M": "AstroPT 850M",
    "clip_base": "CLIP 86M",
    "clip_large": "CLIP 304M",
    "convnext_nano": "ConvNeXt-V2 15M",
    "convnext_tiny": "ConvNeXt-V2 28M",
    "convnext_base": "ConvNeXt-V2 89M",
    "convnext_large": "ConvNeXt-V2 198M",
    "ijepa_huge": "I-JEPA 632M",
    "ijepa_giant": "I-JEPA 1B",
    "llava_15_7b": "LLaVA-1.5 7B",
    "llava_15_13b": "LLaVA-1.5 13B",
    "vit_base": "ViT 86M",
    "vit_large": "ViT 304M",
    "vit_huge": "ViT 632M",
    "vit-mae_base": "ViT-MAE 86M",
    "vit-mae_large": "ViT-MAE 304M",
    "vit-mae_huge": "ViT-MAE 632M",
    "vjepa_large": "V-JEPA-2 300M",
    "vjepa_huge": "V-JEPA-2 600M",
    "vjepa_giant": "V-JEPA-2 1B",
}
FIG_WIDTH_IN = 396 / 72
FIG_HEIGHT_IN = 2.5


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
    name = re.sub(r"_pca\d+$", "", source.removeprefix("single_"))
    return MODEL_LABELS[name]


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

    x = np.arange(len(names))
    ax.plot(x, means, "o", color="#1f77b4", ms=3, linestyle="none", label="single model")

    for src, arr in baskets.items():
        style = BASKET_STYLE[src]
        ax.axhline(arr.mean(), color=style["color"], lw=1.4, label=style["label"])

    best = int(np.argmax(means))
    ax.scatter(
        [best], [means[best]], color="#2ca02c", s=55, marker="*", zorder=5, label="best single"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, ha="center", fontsize=7.5)
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
            f"{modality:<6}"
            f"{cols['basket_mcca_whitened']:>20}"
            f"{cols['basket_concat_pca']:>20}"
            f"{means[best_name]:>14.3f} ({best_name}) "
            f"{median_single:>13.3f}"
            f"{ranks['basket_mcca_whitened']:>7d}/{n_src}"
            f"{ranks['basket_concat_pca']:>9d}/{n_src}"
            f"{ps['basket_mcca_whitened']:>10.3g}"
            f"{ps['basket_concat_pca']:>10.3g}"
        )
    return lines


def main() -> None:
    df = pd.read_parquet(DATA / "cosmos_1024.parquet")
    df = df[df["seed"] < 5]
    df = df[df["kind"] == "regression"] if "kind" in df.columns else df
    FIGS.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=False)
    for i, modality in enumerate(MODALITIES):
        ax = axes[i]
        singles, baskets = cell_pivot(df, modality)
        plot_cell(ax, singles, baskets)
        ax.set_title(MODALITY_LABEL[modality], fontsize=9)
        if i == 0:
            ax.set_ylabel(r"$R^2$")
    handles, labels = _dedup_legend(axes.ravel())
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    pdf_path = FIGS / "basket_vs_singles_pooled_cosmos1024.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Wrote {pdf_path}")

    lines = stats_lines(df)
    txt_path = FIGS / "basket_vs_singles_pooled_cosmos1024.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {txt_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
