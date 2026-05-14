"""Strip-plot of single-model R²s with horizontal bands for each basket source."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon

PROPERTIES = ["redshift", "mass", "sSFR"]
PROPERTY_LABELS = {"redshift": "z", "mass": r"$\log M_\star$", "sSFR": "sSFR"}
MODALITIES = ["hsc", "jwst"]

BASKET_SOURCES = ("basket_mcca_whitened", "basket_concat_pca")
BASKET_STYLE = {
    "basket_mcca_whitened":  dict(color="#ff7f0e", label="MCCA (full-rank whitened)"),
    "basket_concat_pca":     dict(color="#17becf", label="concat → PCA-to-D"),
}


def basket_pivot(df: pl.DataFrame, modality: str, prop: str):
    sub = df.filter((pl.col("modality") == modality) & (pl.col("property") == prop))
    baskets: dict[str, np.ndarray] = {}
    for bsrc in BASKET_SOURCES:
        arr = sub.filter(pl.col("source") == bsrc).sort("seed")["r2"].to_numpy()
        if len(arr):
            baskets[bsrc] = arr
    singles: dict[str, np.ndarray] = {}
    for src in sub["source"].unique().to_list():
        if not src.startswith("single_"):
            continue
        s = sub.filter(pl.col("source") == src).sort("seed")
        name = src.removeprefix("single_").rsplit("_pca", 1)[0]
        singles[name] = s["r2"].to_numpy()
    return singles, baskets


def plot_one_panel(ax, single_pca: dict, baskets: dict, title: str):
    names_sorted = sorted(single_pca.keys(), key=lambda n: single_pca[n].mean())
    means = [single_pca[n].mean() for n in names_sorted]
    stds = [single_pca[n].std() for n in names_sorted]

    x = np.arange(len(names_sorted))
    ax.errorbar(x, means, yerr=stds, fmt="o", color="#1f77b4",
                ms=5, capsize=2, capthick=0.8, elinewidth=0.8,
                label="single (PCA)")

    for bsrc, arr in baskets.items():
        bm, bs_ = arr.mean(), arr.std()
        style = BASKET_STYLE.get(bsrc, {"color": "k", "label": bsrc})
        ax.axhspan(bm - bs_, bm + bs_, color=style["color"], alpha=0.13, zorder=0)
        ax.axhline(bm, color=style["color"], lw=1.5,
                   label=f"{style['label']} = {bm:.3f} ± {bs_:.3f}")

    best_idx = int(np.argmax(means))
    median_val = float(np.median(means))
    ax.scatter([best_idx], [means[best_idx]], color="#2ca02c", s=80, marker="*",
               zorder=5,
               label=f"best single = {means[best_idx]:.3f} ({names_sorted[best_idx]})")
    ax.axhline(median_val, color="#7f7f7f", lw=1.0, ls="--",
               label=f"median single = {median_val:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("$R^2$")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)


def render_plots(data: Path, figs_dir: Path, suffix: str = "") -> None:
    figs_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(data)
    print(f"Loaded {len(df)} rows from {data}")

    stats_lines = [
        f"{'modality':<6}{'property':<10}"
        f"{'mcca_whitened':>22}{'concat_pca':>22}"
        f"{'best_single':>22}{'median_single':>16}"
        f"{'rank_white':>12}{'rank_concat':>12}"
        f"{'p_white':>12}{'p_concat':>12}"
    ]

    pdf_path = figs_dir / f"basket_vs_singles{suffix}.pdf"
    with PdfPages(pdf_path) as pdf:
        for modality in MODALITIES:
            for prop in PROPERTIES:
                singles, baskets = basket_pivot(df, modality, prop)
                means = {n: arr.mean() for n, arr in singles.items()}
                best_name = max(means, key=means.get)
                best_arr = singles[best_name]
                median_single = float(np.median(list(means.values())))

                def _rank_p(b: np.ndarray) -> tuple[int, float]:
                    rank = sum(1 for v in means.values() if v > b.mean()) + 1
                    if len(b) > 0:
                        _, p = wilcoxon(b, best_arr, alternative="two-sided",
                                        zero_method="wilcox")
                    else:
                        p = float("nan")
                    return rank, p

                whitened = baskets.get("basket_mcca_whitened")
                concat = baskets.get("basket_concat_pca")
                rank_w, p_w = _rank_p(whitened) if whitened is not None else (-1, float("nan"))
                rank_c, p_c = _rank_p(concat) if concat is not None else (-1, float("nan"))

                def _fmt(arr):
                    if arr is None:
                        return f"{'—':>14}        "
                    return f"{arr.mean():>14.3f}±{arr.std():.3f}  "

                stats_lines.append(
                    f"{modality:<6}{prop:<10}"
                    f"{_fmt(whitened)}{_fmt(concat)}"
                    f"{means[best_name]:>14.3f} ({best_name[:8]:<8}) "
                    f"{median_single:>14.3f}  "
                    f"{rank_w:>4d}/{len(singles)+1}  "
                    f"{rank_c:>4d}/{len(singles)+1}  "
                    f"{p_w:>10.3g}  {p_c:>10.3g}"
                )

                fig, ax = plt.subplots(figsize=(12, 5.5))
                plot_one_panel(
                    ax, singles, baskets,
                    title=(f"{modality.upper()} — {PROPERTY_LABELS[prop]}: "
                           f"basket vs single models "
                           f"(mcca-whitened {rank_w}/{len(singles)+1}, "
                           f"concat-pca {rank_c}/{len(singles)+1})"),
                )
                ax.legend(loc="lower right", fontsize=7)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    print(f"Wrote {pdf_path}")

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharey=False, sharex=False)
    for i, modality in enumerate(MODALITIES):
        for j, prop in enumerate(PROPERTIES):
            singles, baskets = basket_pivot(df, modality, prop)
            plot_one_panel(
                axes[i, j], singles, baskets,
                title=f"{modality.upper()} — {PROPERTY_LABELS[prop]}",
            )
            if i == 0 and j == 0:
                axes[i, j].legend(loc="lower right", fontsize=6)
    fig.suptitle("Basket-mean (MCCA full-rank whitened, concat→PCA) "
                 "vs single-model $R^2$ — 45 000 COSMOS-Web galaxies",
                 fontsize=13)
    fig.tight_layout()
    sum_path = figs_dir / f"basket_vs_singles_summary{suffix}.pdf"
    fig.savefig(sum_path)
    plt.close(fig)
    print(f"Wrote {sum_path}")

    txt_path = figs_dir / f"basket_vs_singles_stats{suffix}.txt"
    txt_path.write_text("\n".join(stats_lines) + "\n")
    print(f"Wrote {txt_path}")
    print("\n" + "\n".join(stats_lines))
