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

    - `k` column → `bazaar bench scaling` (basket-pruning curves).
    - `covariate` column → `bazaar bench dims` (per-dim covariate decomposition).
    - otherwise → basket-vs-singles strip-plot.
    """
    figs_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(data)
    print(f"Loaded {len(df)} rows from {data}")
    if "k" in df.columns:
        render_scaling_plot(df, figs_dir, suffix)
    elif "covariate" in df.columns:
        render_dimensions(df, figs_dir, suffix)
    elif "prop_i" in df.columns:
        render_probes(df, figs_dir, suffix)
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
            if not np.isfinite(best):
                continue
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


# ---------------------------------------------------------------------------
# `bench dims` rendering — polar "spider web" of per-dim covariate R².
# ---------------------------------------------------------------------------
#
# The dims parquet is long-form `(modality, dim, covariate, covariate_group,
# n_valid, r2, D)`. The plot we want shows, for each modality, all 1024 dims
# arranged around a circle (θ = dim_index / 1024 · 2π) with one polar trace
# per covariate, color-grouped by covariate_group. A reader can immediately
# read off which dim ranges carry which kind of signal and which groups
# dominate the radial spikes.

DIMS_GROUP_ORDER: tuple[str, ...] = (
    "physics", "photometry", "color", "morphology", "systematics", "position",
)
DIMS_GROUP_COLORS: dict[str, str] = {
    "physics":     "#d62728",  # red
    "photometry":  "#ff7f0e",  # orange
    "color":       "#1f77b4",  # blue
    "morphology":  "#2ca02c",  # green
    "systematics": "#7f7f7f",  # grey
    "position":    "#9467bd",  # purple
}
DIMS_ARGMAX_THRESHOLD = 0.05


def _dims_pivot(df: pl.DataFrame, modality: str) -> tuple[
    np.ndarray, list[str], dict[str, str]
]:
    """Wide-form (D, C) R² matrix for one modality, plus aligned covariate
    name and group lookups.
    """
    sub = df.filter(pl.col("modality") == modality)
    if sub.is_empty():
        raise ValueError(f"no rows for modality={modality!r}")

    n_dims = sub["dim"].max() + 1
    cov_names = (
        sub.group_by("covariate").agg(pl.col("r2").max().alias("max_r2"))
        .sort("max_r2", descending=True)["covariate"].to_list()
    )
    group_lookup = dict(
        sub.select("covariate", "covariate_group").unique().iter_rows()
    )

    R = np.zeros((n_dims, len(cov_names)), dtype=np.float32)
    cov_idx = {c: i for i, c in enumerate(cov_names)}
    for r in sub.iter_rows(named=True):
        R[r["dim"], cov_idx[r["covariate"]]] = r["r2"]
    return R, cov_names, group_lookup


def _plot_dims_spider(
    ax, R: np.ndarray, cov_names: list[str], group_lookup: dict[str, str],
    *, title: str,
) -> None:
    """One polar pane: 1024 angular positions × one trace per covariate.

    Each trace is alpha-blended at its group's color. Dominant covariates
    appear as outward radial spikes at the dims that encode them.
    """
    n_dims, n_cov = R.shape
    theta = np.linspace(0, 2 * np.pi, n_dims, endpoint=False)

    # Plot one trace per covariate, color-grouped.
    for c_idx, name in enumerate(cov_names):
        grp = group_lookup[name]
        color = DIMS_GROUP_COLORS[grp]
        ax.plot(
            theta, R[:, c_idx],
            color=color, lw=0.6, alpha=0.45,
        )

    # Outer argmax-group ring: a thin annulus just outside the max trace,
    # color-coded by argmax group at each dim, restricted to informative dims.
    max_r2 = R.max(axis=1)
    argmax_cov = R.argmax(axis=1)
    informative = max_r2 > DIMS_ARGMAX_THRESHOLD
    ring_r = max(R.max() * 1.06, 0.05)
    ring_w = max(R.max() * 0.03, 0.005)

    # Bar segments per dim, only where informative.
    bar_theta = theta[informative]
    bar_widths = np.full(bar_theta.shape, 2 * np.pi / n_dims)
    bar_groups = [group_lookup[cov_names[c]] for c in argmax_cov[informative]]
    bar_colors = [DIMS_GROUP_COLORS[g] for g in bar_groups]
    ax.bar(
        bar_theta, np.full(bar_theta.shape, ring_w),
        width=bar_widths, bottom=ring_r,
        color=bar_colors, edgecolor="none", linewidth=0,
    )

    # Cosmetics.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, ring_r + ring_w * 2)
    ax.set_rlabel_position(135)
    ax.set_title(title, pad=20, fontsize=12)

    # Sparse dim labels around the circle.
    label_dims = np.linspace(0, n_dims, 9)[:-1].astype(int)
    ax.set_xticks(2 * np.pi * label_dims / n_dims)
    ax.set_xticklabels([str(d) for d in label_dims], fontsize=8)
    ax.grid(alpha=0.25)


def _dims_group_legend(fig, cov_names: list[str], group_lookup: dict[str, str]) -> None:
    """Legend keyed by covariate group with the per-group covariate count."""
    from matplotlib.lines import Line2D
    counts: dict[str, int] = {}
    for name in cov_names:
        g = group_lookup[name]
        counts[g] = counts.get(g, 0) + 1
    handles = [
        Line2D([0], [0], color=DIMS_GROUP_COLORS[g], lw=3,
               label=f"{g} (n={counts.get(g, 0)})")
        for g in DIMS_GROUP_ORDER if g in counts
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(len(handles), 6), frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )


def _dims_top_covariate_panel(
    ax, R: np.ndarray, cov_names: list[str], group_lookup: dict[str, str],
    *, title: str, top_n: int = 12,
) -> None:
    """A companion bar chart of the top-N covariates by max R² for context."""
    max_r2 = R.max(axis=0)
    order = np.argsort(max_r2)[::-1][:top_n]
    names = [cov_names[i] for i in order]
    vals = max_r2[order]
    colors = [DIMS_GROUP_COLORS[group_lookup[n]] for n in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("max $R^2$ across dims")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, max(0.8, vals.max() * 1.1))
    for yi, v in zip(y, vals):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)


def render_dimensions(df: pl.DataFrame, figs_dir: Path, suffix: str = "") -> None:
    """Render `bench dims` parquet as a per-modality polar spider plot."""
    modalities = df["modality"].unique().to_list()
    modalities.sort()

    # Multi-page PDF: one full-page polar + one companion bar chart per modality.
    pdf_path = figs_dir / f"dims_spider{suffix}.pdf"
    with PdfPages(pdf_path) as pdf:
        for mod in modalities:
            R, cov_names, group_lookup = _dims_pivot(df, mod)

            fig = plt.figure(figsize=(11, 11))
            ax = fig.add_subplot(111, projection="polar")
            _plot_dims_spider(
                ax, R, cov_names, group_lookup,
                title=f"{mod.upper()} — per-dim covariate $R^2$ "
                      f"({R.shape[0]} dims × {R.shape[1]} covariates, "
                      f"outer ring = argmax group where max $R^2$ > "
                      f"{DIMS_ARGMAX_THRESHOLD})",
            )
            _dims_group_legend(fig, cov_names, group_lookup)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5.5))
            _dims_top_covariate_panel(
                ax, R, cov_names, group_lookup,
                title=f"{mod.upper()} — top covariates by max $R^2$",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {pdf_path}")

    # Combined 1×2 polar summary (HSC | JWST) for the headline figure.
    n_mod = len(modalities)
    fig = plt.figure(figsize=(7.5 * n_mod, 8.5))
    cov_names_legend: list[str] | None = None
    group_lookup_legend: dict[str, str] | None = None
    for i, mod in enumerate(modalities):
        R, cov_names, group_lookup = _dims_pivot(df, mod)
        ax = fig.add_subplot(1, n_mod, i + 1, projection="polar")
        _plot_dims_spider(
            ax, R, cov_names, group_lookup,
            title=f"{mod.upper()}",
        )
        cov_names_legend, group_lookup_legend = cov_names, group_lookup
    if cov_names_legend is not None:
        _dims_group_legend(fig, cov_names_legend, group_lookup_legend or {})
    fig.suptitle(
        f"Per-dim covariate $R^2$ — {R.shape[0]} Bazaar dims × "
        f"{R.shape[1]} covariates",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    sum_path = figs_dir / f"dims_spider_summary{suffix}.pdf"
    fig.savefig(sum_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {sum_path}")


# ---------------------------------------------------------------------------
# `bench probes` rendering — 3×3 probe-direction cosine matrices per modality.
# ---------------------------------------------------------------------------
#
# The parquet is long-form (modality, source, prop_i, prop_j, cos, D). Two
# special source names — `basket_mcca_whitened` and `basket_avg` — frame
# Bazaar's S geometry against the basket-elementwise mean. The remaining 22
# `single_<family>_<size>` rows give the per-model spread.

PROBES_PROPERTIES: tuple[str, ...] = ("redshift", "mass", "sSFR")
PROBES_PROPERTY_LABEL: dict[str, str] = {
    "redshift": r"$z$",
    "mass":     r"$\log M_\star$",
    "sSFR":     "sSFR",
}
PROBES_HEADLINE_SOURCES: tuple[str, ...] = ("basket_mcca_whitened", "basket_avg")
PROBES_HEADLINE_LABEL: dict[str, str] = {
    "basket_mcca_whitened": "Bazaar S (MCCA)",
    "basket_avg":           "basket-avg (per-model PCA)",
}


def _probes_pivot(df: pl.DataFrame, modality: str, source: str) -> np.ndarray:
    """Long-form rows → (3, 3) cosine matrix in PROBES_PROPERTIES order."""
    sub = df.filter((pl.col("modality") == modality) & (pl.col("source") == source))
    if sub.is_empty():
        raise ValueError(f"no rows for modality={modality!r}, source={source!r}")
    idx = {p: i for i, p in enumerate(PROBES_PROPERTIES)}
    M = np.full((3, 3), np.nan, dtype=np.float32)
    for r in sub.iter_rows(named=True):
        M[idx[r["prop_i"]], idx[r["prop_j"]]] = r["cos"]
    if np.isnan(M).any():
        raise ValueError(
            f"missing cells in 3×3 for modality={modality!r}, source={source!r}"
        )
    return M


def _probes_singles_stack(df: pl.DataFrame, modality: str) -> np.ndarray:
    """Stack of (n_models, 3, 3) cosine matrices over single_* sources."""
    single_sources = sorted(
        s for s in df["source"].unique().to_list() if s.startswith("single_")
    )
    if not single_sources:
        raise ValueError(f"no single_* sources for modality={modality!r}")
    return np.stack(
        [_probes_pivot(df, modality, s) for s in single_sources], axis=0,
    )


def _plot_probes_heatmap(ax, M: np.ndarray, *, title: str, vmax: float = 1.0) -> None:
    """3×3 cosine heatmap with PROBES_PROPERTIES labels and cell annotations."""
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    labels = [PROBES_PROPERTY_LABEL[p] for p in PROBES_PROPERTIES]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(3):
        for j in range(3):
            val = M[i, j]
            ax.text(
                j, i, f"{val:+.2f}",
                ha="center", va="center",
                color="black" if abs(val) < 0.55 else "white",
                fontsize=10,
            )
    ax.set_title(title, fontsize=11)
    return im


def _plot_probes_spread(ax, df: pl.DataFrame, modality: str) -> None:
    """Off-diagonal spread plot: three pairs × per-model dots + Bazaar/avg lines.

    For each of the three upper-triangle pairs (z–M*, z–sSFR, M*–sSFR), draw
    every basket member's cosine as a faint dot, then mark Bazaar S and the
    basket-avg with horizontal ticks. Highlights whether Bazaar's value lies
    inside the per-model spread.
    """
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_labels = [
        f"{PROBES_PROPERTY_LABEL[PROBES_PROPERTIES[i]]}–"
        f"{PROBES_PROPERTY_LABEL[PROBES_PROPERTIES[j]]}"
        for i, j in pairs
    ]
    singles = _probes_singles_stack(df, modality)            # (n, 3, 3)
    B = _probes_pivot(df, modality, "basket_mcca_whitened")
    A = _probes_pivot(df, modality, "basket_avg")

    rng = np.random.default_rng(0)
    for x, (i, j) in enumerate(pairs):
        vals = singles[:, i, j]
        jitter = rng.uniform(-0.18, 0.18, size=vals.shape)
        ax.scatter(
            x + jitter, vals,
            color="#1f77b4", s=22, alpha=0.55, edgecolor="none",
            label="single model" if x == 0 else None,
        )
        # Bazaar marker.
        ax.scatter(
            [x], [B[i, j]],
            color="#ff7f0e", marker="D", s=110, zorder=5, edgecolor="black",
            linewidth=0.8,
            label="Bazaar S" if x == 0 else None,
        )
        # basket-avg marker.
        ax.scatter(
            [x], [A[i, j]],
            color="#17becf", marker="*", s=180, zorder=4, edgecolor="black",
            linewidth=0.7,
            label="basket-avg" if x == 0 else None,
        )

    ax.axhline(0.0, color="#aaaaaa", lw=0.7, zorder=0)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_labels, fontsize=10)
    ax.set_ylabel(r"$\cos(\mathbf{w}_a, \mathbf{w}_b)$")
    ax.set_title(f"{modality.upper()} — probe-direction cosines by pair", fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=8, frameon=True)


def _probes_stats_lines(df: pl.DataFrame) -> list[str]:
    """Tabulate per-modality off-diagonals + in-spread flags."""
    lines = [
        f"{'modality':<10}{'pair':<10}"
        f"{'bazaar':>9}{'bask_avg':>10}{'bask_med':>10}"
        f"{'min':>9}{'max':>9}{'in?':>6}"
    ]
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_lbl = ["z-M*", "z-sSFR", "M*-sSFR"]
    for mod in sorted(df["modality"].unique().to_list()):
        B = _probes_pivot(df, mod, "basket_mcca_whitened")
        A = _probes_pivot(df, mod, "basket_avg")
        S = _probes_singles_stack(df, mod)
        for (i, j), lbl in zip(pairs, pair_lbl):
            sp = S[:, i, j]
            lo, hi = float(sp.min()), float(sp.max())
            med = float(np.median(sp))
            b = float(B[i, j])
            inside = "yes" if lo <= b <= hi else "NO"
            lines.append(
                f"{mod:<10}{lbl:<10}"
                f"{b:>+9.3f}{float(A[i, j]):>+10.3f}{med:>+10.3f}"
                f"{lo:>+9.3f}{hi:>+9.3f}{inside:>6}"
            )
    return lines


def render_probes(df: pl.DataFrame, figs_dir: Path, suffix: str = "") -> None:
    """Render `bench probes` parquet: per-modality heatmaps + spread plot."""
    modalities = sorted(df["modality"].unique().to_list())

    # Multi-page PDF: per modality, heatmaps + off-diagonal spread.
    pdf_path = figs_dir / f"probes_panels{suffix}.pdf"
    with PdfPages(pdf_path) as pdf:
        for mod in modalities:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
            for ax, src in zip(axes, PROBES_HEADLINE_SOURCES):
                M = _probes_pivot(df, mod, src)
                _plot_probes_heatmap(
                    ax, M,
                    title=f"{mod.upper()} — {PROBES_HEADLINE_LABEL[src]}",
                )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5.5))
            _plot_probes_spread(ax, df, mod)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {pdf_path}")

    # 2×2 summary heatmap: modalities × {Bazaar S, basket-avg}.
    n_rows = len(modalities)
    n_cols = len(PROBES_HEADLINE_SOURCES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )
    last_im = None
    for i, mod in enumerate(modalities):
        for j, src in enumerate(PROBES_HEADLINE_SOURCES):
            M = _probes_pivot(df, mod, src)
            last_im = _plot_probes_heatmap(
                axes[i, j], M,
                title=f"{mod.upper()} — {PROBES_HEADLINE_LABEL[src]}",
            )
    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=axes.ravel().tolist(),
            fraction=0.025, pad=0.04,
        )
        cbar.set_label(r"$\cos(\mathbf{w}_a, \mathbf{w}_b)$")
    fig.suptitle(
        "Probe-direction cosine matrices — Bazaar S vs basket-avg (PU §3.1 Fig 4)",
        fontsize=13,
    )
    sum_path = figs_dir / f"probes_summary{suffix}.pdf"
    fig.savefig(sum_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {sum_path}")

    # Spread-only summary (off-diagonals per modality side-by-side).
    fig, axes = plt.subplots(
        1, n_rows, figsize=(7.5 * n_rows, 5.0), squeeze=False,
    )
    for i, mod in enumerate(modalities):
        _plot_probes_spread(axes[0, i], df, mod)
    fig.suptitle(
        "Probe-direction off-diagonals — basket spread vs Bazaar S vs basket-avg",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    spread_path = figs_dir / f"probes_spread{suffix}.pdf"
    fig.savefig(spread_path)
    plt.close(fig)
    print(f"Wrote {spread_path}")

    # Stats text companion.
    stats_lines = _probes_stats_lines(df)
    txt_path = figs_dir / f"probes_stats{suffix}.txt"
    txt_path.write_text("\n".join(stats_lines) + "\n")
    print(f"Wrote {txt_path}")
    print("\n" + "\n".join(stats_lines))
