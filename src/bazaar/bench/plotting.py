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

BASKET_SOURCES = ("basket_mean", "basket_procrustes_mean", "basket_mcca_mean")
BASKET_STYLE = {
    "basket_mean":             dict(color="#d62728", label="naive mean"),
    "basket_procrustes_mean":  dict(color="#9467bd", label="Procrustes-aligned mean"),
    "basket_mcca_mean":        dict(color="#ff7f0e", label="MCCA shared latent"),
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
    ax.scatter([best_idx], [means[best_idx]], color="#2ca02c", s=80, marker="*",
               zorder=5,
               label=f"best single = {means[best_idx]:.3f} ({names_sorted[best_idx]})")

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
        f"{'naive_mean':>16}{'procrustes_mean':>22}{'mcca_mean':>22}"
        f"{'best_single':>22}"
        f"{'rank_naive':>12}{'rank_proc':>12}{'rank_mcca':>12}"
        f"{'p_naive':>12}{'p_proc':>12}{'p_mcca':>12}"
    ]

    pdf_path = figs_dir / f"basket_vs_singles{suffix}.pdf"
    with PdfPages(pdf_path) as pdf:
        for modality in MODALITIES:
            for prop in PROPERTIES:
                singles, baskets = basket_pivot(df, modality, prop)
                means = {n: arr.mean() for n, arr in singles.items()}
                best_name = max(means, key=means.get)
                best_arr = singles[best_name]

                def _rank_p(b: np.ndarray) -> tuple[int, float]:
                    rank = sum(1 for v in means.values() if v > b.mean()) + 1
                    if len(b) > 0:
                        _, p = wilcoxon(b, best_arr, alternative="two-sided",
                                        zero_method="wilcox")
                    else:
                        p = float("nan")
                    return rank, p

                naive = baskets.get("basket_mean")
                proc = baskets.get("basket_procrustes_mean")
                mcca = baskets.get("basket_mcca_mean")
                rank_n, p_n = _rank_p(naive) if naive is not None else (-1, float("nan"))
                rank_p, p_p = _rank_p(proc) if proc is not None else (-1, float("nan"))
                rank_m, p_m = _rank_p(mcca) if mcca is not None else (-1, float("nan"))

                def _fmt(arr):
                    if arr is None:
                        return f"{'—':>14}        "
                    return f"{arr.mean():>14.3f}±{arr.std():.3f}  "

                stats_lines.append(
                    f"{modality:<6}{prop:<10}"
                    f"{_fmt(naive)}{_fmt(proc)}{_fmt(mcca)}"
                    f"{means[best_name]:>14.3f} ({best_name[:8]:<8}) "
                    f"{rank_n:>4d}/{len(singles)+1}  "
                    f"{rank_p:>4d}/{len(singles)+1}  "
                    f"{rank_m:>4d}/{len(singles)+1}  "
                    f"{p_n:>10.3g}  {p_p:>10.3g}  {p_m:>10.3g}"
                )

                fig, ax = plt.subplots(figsize=(12, 5.5))
                plot_one_panel(
                    ax, singles, baskets,
                    title=(f"{modality.upper()} — {PROPERTY_LABELS[prop]}: "
                           f"basket vs single models "
                           f"(naive {rank_n}/{len(singles)+1}, "
                           f"procrustes {rank_p}/{len(singles)+1}, "
                           f"mcca {rank_m}/{len(singles)+1})"),
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
    fig.suptitle("Basket-mean (naive, Procrustes-aligned, MCCA) "
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
