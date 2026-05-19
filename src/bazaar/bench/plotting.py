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
    """Top-level renderer; dispatches by parquet schema.

    Parquets carrying a `k` column come from `bazaar bench scaling` and are
    routed to the basket-pruning curve renderer; everything else is the
    basket-vs-singles strip-plot.
    """
    figs_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(data)
    print(f"Loaded {len(df)} rows from {data}")
    if "k" in df.columns:
        render_scaling_plot(df, figs_dir, suffix)
    else:
        render_basket_vs_singles_plot(df, figs_dir, suffix)


def render_basket_vs_singles_plot(
    df: pl.DataFrame, figs_dir: Path, suffix: str = "",
) -> None:
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


# ---------------------------------------------------------------------------
# Scaling plot (basket pruning curves)
# ---------------------------------------------------------------------------

SCALING_KINDS = ("random", "one_per_family")
SCALING_KIND_STYLE = {
    "random":         dict(marker="o", linestyle="-",  label_suffix="random"),
    "one_per_family": dict(marker="s", linestyle="--", label_suffix="one/family"),
}


def _scaling_pivot(
    df: pl.DataFrame, modality: str, prop: str, source: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For one (modality, property, source), build {subset_kind: (ks, means, stds)}.

    Means and stds are taken over the joint (subset_id, seed) population at
    each k. Only `random` and `one_per_family` are returned; the `full` row
    at k = max(ks) is folded into both series so the curves terminate cleanly.
    """
    sub = df.filter(
        (pl.col("modality") == modality)
        & (pl.col("property") == prop)
        & (pl.col("source") == source)
    )
    if sub.is_empty():
        return {}

    full_rows = sub.filter(pl.col("subset_kind") == "full")
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for kind in SCALING_KINDS:
        kind_rows = sub.filter(pl.col("subset_kind") == kind)
        if kind_rows.is_empty() and full_rows.is_empty():
            continue
        joined = pl.concat([kind_rows, full_rows]) if not full_rows.is_empty() else kind_rows
        agg = (
            joined.group_by("k")
            .agg([
                pl.col("r2").mean().alias("mean"),
                pl.col("r2").std(ddof=0).fill_null(0.0).alias("std"),
            ])
            .sort("k")
        )
        ks = agg["k"].to_numpy()
        means = agg["mean"].to_numpy()
        stds = agg["std"].to_numpy()
        out[kind] = (ks, means, stds)
    return out


def _singles_ref(df: pl.DataFrame, modality: str, prop: str) -> tuple[float, float]:
    """(best_single_mean, median_single_mean) across all single-model rows."""
    singles = df.filter(
        (pl.col("modality") == modality)
        & (pl.col("property") == prop)
        & (pl.col("source").str.starts_with("single_"))
    )
    if singles.is_empty():
        return float("nan"), float("nan")
    per_model = (
        singles.group_by("source")
        .agg(pl.col("r2").mean().alias("mean"))
    )
    means = per_model["mean"].to_numpy()
    return float(means.max()), float(np.median(means))


def _plot_scaling_panel(
    ax, df: pl.DataFrame, modality: str, prop: str, title: str,
) -> None:
    any_curve = False
    for source in BASKET_SOURCES:
        per_kind = _scaling_pivot(df, modality, prop, source)
        style = BASKET_STYLE.get(source, {"color": "k", "label": source})
        for kind, (ks, means, stds) in per_kind.items():
            ks_style = SCALING_KIND_STYLE[kind]
            label = f"{style['label']} ({ks_style['label_suffix']})"
            ax.errorbar(
                ks, means, yerr=stds,
                color=style["color"],
                marker=ks_style["marker"], linestyle=ks_style["linestyle"],
                ms=5, capsize=2, capthick=0.7, elinewidth=0.7, lw=1.2,
                label=label,
            )
            any_curve = True

    best, median = _singles_ref(df, modality, prop)
    if np.isfinite(best):
        ax.axhline(best, color="#2ca02c", lw=1.0, ls=":",
                   label=f"best single = {best:.3f}")
    if np.isfinite(median):
        ax.axhline(median, color="#7f7f7f", lw=1.0, ls="--",
                   label=f"median single = {median:.3f}")

    ax.set_xlabel("basket size $k$")
    ax.set_ylabel("$R^2$")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    if any_curve:
        ks_all = sorted({
            int(x) for x in df.filter(pl.col("k").is_not_null())["k"].unique()
        })
        ax.set_xticks(ks_all)


def render_scaling_plot(
    df: pl.DataFrame, figs_dir: Path, suffix: str = "",
) -> None:
    """Per-cell scaling curves: R² vs basket size k, random vs one-per-family."""
    modalities = sorted(df["modality"].unique().to_list())
    properties = [p for p in PROPERTIES if p in df["property"].unique().to_list()]

    stats_lines = [
        f"{'modality':<6}{'property':<10}{'source':<22}"
        f"{'kind':<16}{'k':>4}{'mean_r2':>12}{'std_r2':>10}{'n':>5}"
    ]
    long_rows = (
        df.filter(pl.col("k").is_not_null())
        .group_by(["modality", "property", "source", "subset_kind", "k"])
        .agg([
            pl.col("r2").mean().alias("mean_r2"),
            pl.col("r2").std(ddof=0).fill_null(0.0).alias("std_r2"),
            pl.col("r2").len().alias("n"),
        ])
        .sort(["modality", "property", "source", "subset_kind", "k"])
    )
    for row in long_rows.iter_rows(named=True):
        stats_lines.append(
            f"{row['modality']:<6}{row['property']:<10}"
            f"{row['source']:<22}{row['subset_kind']:<16}"
            f"{int(row['k']):>4d}{row['mean_r2']:>12.4f}"
            f"{row['std_r2']:>10.4f}{int(row['n']):>5d}"
        )
    for modality in modalities:
        for prop in properties:
            best, median = _singles_ref(df, modality, prop)
            stats_lines.append(
                f"{modality:<6}{prop:<10}{'best_single':<22}"
                f"{'reference':<16}{'-':>4}{best:>12.4f}{0.0:>10.4f}{0:>5d}"
            )
            stats_lines.append(
                f"{modality:<6}{prop:<10}{'median_single':<22}"
                f"{'reference':<16}{'-':>4}{median:>12.4f}{0.0:>10.4f}{0:>5d}"
            )

    pdf_path = figs_dir / f"scaling_curves{suffix}.pdf"
    with PdfPages(pdf_path) as pdf:
        for modality in modalities:
            for prop in properties:
                fig, ax = plt.subplots(figsize=(8, 5.5))
                _plot_scaling_panel(
                    ax, df, modality, prop,
                    title=(f"{modality.upper()} — {PROPERTY_LABELS.get(prop, prop)}: "
                           f"basket size scaling"),
                )
                ax.legend(loc="lower right", fontsize=7)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    print(f"Wrote {pdf_path}")

    n_rows, n_cols = len(modalities), len(properties)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4.5 * n_rows),
        sharey=False, sharex=False, squeeze=False,
    )
    for i, modality in enumerate(modalities):
        for j, prop in enumerate(properties):
            _plot_scaling_panel(
                axes[i, j], df, modality, prop,
                title=f"{modality.upper()} — {PROPERTY_LABELS.get(prop, prop)}",
            )
            if i == 0 and j == 0:
                axes[i, j].legend(loc="lower right", fontsize=6)
    fig.suptitle(
        "Basket pruning curves — $R^2$ vs basket size $k$, "
        "random vs one-per-family",
        fontsize=13,
    )
    fig.tight_layout()
    sum_path = figs_dir / f"scaling_curves_summary{suffix}.pdf"
    fig.savefig(sum_path)
    plt.close(fig)
    print(f"Wrote {sum_path}")

    txt_path = figs_dir / f"scaling_curves_stats{suffix}.txt"
    txt_path.write_text("\n".join(stats_lines) + "\n")
    print(f"Wrote {txt_path}")
