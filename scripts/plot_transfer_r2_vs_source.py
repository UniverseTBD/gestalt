#!/usr/bin/env python3
"""Source-to-target transfer heatmaps for `bench transfer`.

The main PDF contains two coloured square 4×4 matrices: one each for
MAX-VAR GCCA and concat→PCA. Rows are the four target metrics (including GZ10
morphology); columns are the four corpora used for the representation fit.
Each transfer cell shows the mean ± standard deviation across five probe
seeds, with target fit cells outlined. A separate PDF contains the rotated
raw-score comparison for Figure 2d.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "DejaVu Sans"],
    },
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGS = REPO / "assets" / "plots"
SUMMARY = REPO.parent / "table1_main_comparison.csv"
FIG_WIDTH_IN = 396 / 72

FIT_SOURCES = ["cosmos-hsc", "cosmos-jwst", "gz10", "galaxies"]
TARGETS = ["cosmos-hsc", "cosmos-jwst", "gz10", "galaxies"]
TARGET_LABEL = {
    "cosmos-hsc": "COSMOS-Web (HSC)",
    "cosmos-jwst": "COSMOS-Web (JWST)",
    "gz10": "GZ10",
    "galaxies": "Smith42/galaxies",
}
FIT_SOURCE_LABEL = {
    "cosmos-hsc": "HSC",
    "cosmos-jwst": "JWST",
    "gz10": "GZ10",
    "galaxies": "Galaxies",
}

SOURCE_STYLE = {
    "basket_mcca_whitened": {
        "label": "Gestalt",
        "color": "#ff7f0e",
        "marker": "o",
        "offset": -0.20,
    },
    "basket_concat_pca": {
        "label": r"Basket (concat$\to$PCA)",
        "color": "#17becf",
        "marker": "s",
        "offset": 0.00,
    },
    "single_astropt_850M_pca_zscore": {
        "label": "AstroPT 850M (single)",
        "color": "#e377c2",
        "marker": "D",
        "offset": 0.20,
    },
}
SOURCE_ORDER = list(SOURCE_STYLE.keys())

# Galaxies per-property page (matches plot_r2_vs_params_galaxies.py).
GALAXIES_PROPERTIES = [
    "artifact",
    "disc",
    "edge_on",
    "g_minus_r",
    "log_mstar",
    "mag_abs_g",
    "mag_abs_z",
    "mean_ssfr",
    "photo_z",
    "r_minus_z",
    "smooth",
    "spec_z",
    "tight_spiral",
]
PROPERTY_LABEL = {
    "redshift": r"$z_{\rm phot}$",
    "mass": r"$\log M_\star$",
    "sSFR": r"sSFR",
    "gz10_label": r"GZ10 class",
    "photo_z": r"$z_{\rm phot}$",
    "spec_z": r"$z_{\rm spec}$",
    "mag_abs_g": r"$M_g$",
    "mag_abs_z": r"$M_z$",
    "g_minus_r": r"$g{-}r$",
    "r_minus_z": r"$r{-}z$",
    "log_mstar": r"$\log M_\star$",
    "mean_ssfr": r"sSFR",
    "smooth": r"smooth",
    "disc": r"disc",
    "artifact": r"artifact",
    "edge_on": r"edge-on",
    "tight_spiral": r"tight spiral",
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


def load_all_transfer() -> pd.DataFrame:
    parts = []
    for target in TARGETS:
        p = DATA / f"transfer_{target}.parquet"
        if not p.exists():
            print(f"warning: missing {p}")
            continue
        parts.append(pd.read_parquet(p))
    if not parts:
        raise FileNotFoundError("no transfer_*.parquet found in data/")
    return pd.concat(parts, ignore_index=True)


def _plot_strip(
    ax,
    panel: pd.DataFrame,
    target: str,
    *,
    metric: str = "r2",
) -> None:
    """One target panel: jittered strip per (fit_source, source)."""
    rng = np.random.default_rng(0)
    width = 0.14

    for src in SOURCE_ORDER:
        style = SOURCE_STYLE[src]
        rows = cast(pd.DataFrame, panel[panel["source"] == src])
        if rows.empty:
            continue
        for i, fs in enumerate(FIT_SOURCES):
            vals = np.asarray(
                rows.loc[rows["fit_source"] == fs, metric],
                dtype=float,
            )
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            x_center = i + style["offset"]
            jitter = rng.uniform(-width, width, size=vals.size)
            ax.scatter(
                x_center + jitter,
                vals,
                color=style["color"],
                marker=style["marker"],
                s=22,
                alpha=0.65,
                edgecolors="black",
                linewidths=0.3,
                label=style["label"] if i == 0 else None,
            )
            # Mean tick at the per-(src, fs) cluster center.
            ax.plot(
                [x_center - width, x_center + width],
                [vals.mean(), vals.mean()],
                color=style["color"],
                lw=1.6,
                solid_capstyle="butt",
            )

    # On-domain reference: MCCA mean at fit_source == target.
    on_dom = np.asarray(
        panel.loc[
            (panel["source"] == "basket_mcca_whitened") & (panel["fit_source"] == target),
            metric,
        ],
        dtype=float,
    )
    on_dom = on_dom[np.isfinite(on_dom)]
    if on_dom.size > 0:
        ax.axhline(
            on_dom.mean(),
            color=SOURCE_STYLE["basket_mcca_whitened"]["color"],
            lw=1.0,
            ls=":",
            alpha=0.85,
            label="on-domain MCCA mean",
        )

    # Δ = on-domain − off-domain MCCA mean.
    off_dom = np.asarray(
        panel.loc[
            (panel["source"] == "basket_mcca_whitened") & (panel["fit_source"] != target),
            metric,
        ],
        dtype=float,
    )
    off_dom = off_dom[np.isfinite(off_dom)]
    if on_dom.size > 0 and off_dom.size > 0:
        delta = on_dom.mean() - off_dom.mean()
        ax.text(
            0.95,
            0.02,
            f"Δ = {delta:+.3f}",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=9,
        )

    ax.set_xticks(range(len(FIT_SOURCES)))
    tick_labels = []
    for fs in FIT_SOURCES:
        label = FIT_SOURCE_LABEL[fs]
        if fs == target:
            tick_labels.append(f"{label}\n(on-dom.)")
        else:
            tick_labels.append(label)
    ax.set_xticklabels(tick_labels, fontsize=8)
    # Bold the on-domain tick label.
    for tl, fs in zip(ax.get_xticklabels(), FIT_SOURCES):
        if fs == target:
            tl.set_fontweight("bold")
    _inward_ticks(ax)


HEATMAP_SOURCES = ("basket_mcca_whitened", "basket_concat_pca")
LODO_STRATEGIES = (
    "LOSO MCCA-7.5k (2.5k×3)",
    "LOSO MCCA-30k (10k×3)",
)
LODO_ROWS = ("COSMOS-Web HSC", "COSMOS-Web JWST", "GZ10", r"$\mathtt{Smith42/galaxies}$")
BEST_NATIVE_SINGLE = {
    "cosmos-hsc": "single_vjepa_giant_native",
    "cosmos-jwst": "single_vjepa_giant_native",
    "gz10": "single_clip_large_native",
    "galaxies": "single_clip_large_native",
}
COMPARISON_LABELS = (
    "Single-source Gestalt transfer",
    "Cross-survey Gestalt fit",
    "Best-performing model per survey",
)

HEATMAP_ROWS = [
    ("cosmos-hsc", "regression", "r2", "COSMOS-Web HSC"),
    ("cosmos-jwst", "regression", "r2", "COSMOS-Web JWST"),
    ("gz10", "classification", "f1", "GZ10 morphology"),
    ("galaxies", "regression", "r2", "Smith42/galaxies"),
]


def _summary_matrix(
    df: pd.DataFrame,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean/std cells for one representation across the four rows."""
    means: list[list[float]] = []
    stds: list[list[float]] = []
    for target, kind, metric, _ in HEATMAP_ROWS:
        panel = cast(
            pd.DataFrame,
            df[(df["target"] == target) & (df["kind"] == kind) & (df["source"] == source)],
        )
        per_seed = cast(
            pd.DataFrame,
            panel.groupby(["fit_source", "seed"], as_index=False)[metric].mean(),
        )
        summary = cast(
            pd.DataFrame,
            per_seed.groupby("fit_source")[metric].agg(["mean", "std"]),
        )
        ordered = summary.reindex(FIT_SOURCES)
        means.append(np.asarray(ordered["mean"], dtype=float).tolist())
        stds.append(np.asarray(ordered["std"], dtype=float).tolist())
    return np.asarray(means), np.asarray(stds)


def _lodo_summary_matrix() -> np.ndarray:
    """Read the rounded Figure 4 corpus-level summary values."""
    if not SUMMARY.exists():
        raise FileNotFoundError(f"missing Figure 4 summary: {SUMMARY}")
    table = pd.read_csv(SUMMARY).set_index("strategy")
    selected = table.reindex(LODO_STRATEGIES)
    columns = [f"{target}_mean" for target in FIT_SOURCES]
    return np.asarray(selected[columns], dtype=float).T


def _source_task_mean(panel: pd.DataFrame, source: str, metric: str) -> float:
    """Average one source's task means across the five native seeds."""
    values = np.asarray(
        panel.loc[panel["source"] == source].groupby("seed")[metric].mean(),
        dtype=float,
    )
    return values.mean()


def _cross_survey_summary_vector(df: pd.DataFrame) -> np.ndarray:
    """Return equal-task means across the three off-diagonal source fits."""
    means = []
    for target in TARGETS:
        panel = cast(
            pd.DataFrame,
            df[
                (df["target"] == target)
                & (df["fit_source"] != target)
                & (df["source"] == "basket_mcca_whitened")
            ],
        ).copy()
        if target == "gz10":
            panel["score"] = np.where(panel["kind"] == "classification", panel["f1"], panel["r2"])
            metric = "score"
        else:
            metric = "r2"
        per_source_seed = panel.groupby(["fit_source", "seed"])[metric].mean()
        means.append(np.asarray(per_source_seed, dtype=float).mean())
    return np.asarray(means)


def _lodo_7k5_summary_vector() -> np.ndarray:
    """Return the rounded Figure 4 7.5k LODO means."""
    return _lodo_summary_matrix()[:, LODO_STRATEGIES.index("LOSO MCCA-7.5k (2.5k×3)")]


def _native_single_summary_vector() -> np.ndarray:
    """Return fixed best-single means using equal task weighting."""
    cosmos = pd.read_parquet(DATA / "results_native_cosmos.parquet")
    cosmos = cosmos[cosmos["D"] == 1024]
    means = []
    for modality, target in (("hsc", "cosmos-hsc"), ("jwst", "cosmos-jwst")):
        panel = cast(pd.DataFrame, cosmos[cosmos["modality"] == modality])
        means.append(_source_task_mean(panel, BEST_NATIVE_SINGLE[target], "r2"))

    gz10 = pd.read_parquet(DATA / "results_native_gz10.parquet")
    gz10 = cast(pd.DataFrame, gz10[gz10["D"] == 1024].copy())
    gz10["score"] = np.where(gz10["kind"] == "classification", gz10["f1"], gz10["r2"])
    means.append(_source_task_mean(gz10, BEST_NATIVE_SINGLE["gz10"], "score"))

    galaxies = pd.read_parquet(DATA / "results_native_galaxies.parquet")
    galaxies = cast(pd.DataFrame, galaxies[galaxies["D"] == 1024])
    means.append(_source_task_mean(galaxies, BEST_NATIVE_SINGLE["galaxies"], "r2"))
    return np.asarray(means)


def _comparison_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return native, LODO, and best-single score columns."""
    return np.column_stack(
        (
            _cross_survey_summary_vector(df),
            _lodo_7k5_summary_vector(),
            _native_single_summary_vector(),
        ),
    )


def plot_mean_grid(df: pd.DataFrame) -> None:
    """Write the two source-transfer heatmaps."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.6, 4.5),
        sharey=False,
        constrained_layout=True,
    )
    vmin, vmax = 0.15, 0.80
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "viridis"
    row_labels = [row[3] for row in HEATMAP_ROWS]
    col_labels = [FIT_SOURCE_LABEL[fit] for fit in FIT_SOURCES]
    for ax, source in zip(axes[:2], HEATMAP_SOURCES):
        means, stds = _summary_matrix(df, source)
        ax.imshow(means, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(len(FIT_SOURCES)), col_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(row_labels)), row_labels)
        ax.tick_params(labelsize=8)
        ax.set_xlabel(SOURCE_STYLE[source]["label"], fontsize=10, labelpad=10)
        for row, target in enumerate([item[0] for item in HEATMAP_ROWS]):
            col = FIT_SOURCES.index(target)
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.4,
                ),
            )
        for row in range(means.shape[0]):
            for col in range(means.shape[1]):
                value = means[row, col]
                text = f"{value:.3f}\n±{stds[row, col]:.3f}"
                color = "white" if norm(value) > 0.58 else "black"
                ax.text(col, row, text, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes[:2],
        fraction=0.025,
        pad=0.02,
        label="Score ($R^2$ or macro-$F_1$)",
    )
    out = FIGS / "transfer_r2_vs_source.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_comparison_strip(df: pd.DataFrame) -> None:
    """Write the compact, rotated raw-score comparison for Figure 2d."""
    comparison = _comparison_matrix(df)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 1.8), constrained_layout=True)
    x = np.arange(len(LODO_ROWS), dtype=float)
    colors = ("#ff7f0e", "#17becf", "#2ca02c")
    markers = ("o", "s", "*")
    offsets = (-0.18, 0.0, 0.18)
    for col, (label, color, marker, offset) in enumerate(
        zip(COMPARISON_LABELS, colors, markers, offsets),
    ):
        ax.scatter(
            x + offset,
            comparison[:, col],
            color=color,
            marker=marker,
            s=34,
            edgecolors="black",
            linewidths=0.4,
            label=label,
            zorder=2,
        )
        for index, value in enumerate(comparison[:, col]):
            ax.text(
                x[index] + offset,
                value + 0.020,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    ax.set_xticks(x, LODO_ROWS, rotation=0, ha="center")
    ax.set_ylim(0.30, 0.85)
    ax.set_ylabel(r"$R^2$/macro-$F_1$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        columnspacing=0.5,
        handletextpad=0.25,
    )
    out = FIGS / "transfer_native_lodo_single_strip.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def _per_property_page(
    df: pd.DataFrame,
    target: str,
    properties: list[str],
    *,
    figsize: tuple[float, float],
    ncols: int,
    nrows: int,
) -> Figure:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    panel_df = df[(df["target"] == target) & (df["kind"] == "regression")]
    flat = list(axes.flat)
    for idx, prop in enumerate(properties):
        ax = flat[idx]
        sub = cast(pd.DataFrame, panel_df[panel_df["property"] == prop])
        _plot_strip(ax, sub, target)
        ax.set_title(PROPERTY_LABEL.get(prop, prop), fontsize=10)
        if idx % ncols == 0:
            ax.set_ylabel(r"$R^2$", fontsize=9)
        ax.tick_params(labelsize=7)
    for idx in range(len(properties), nrows * ncols):
        flat[idx].set_visible(False)

    fig.suptitle(f"{TARGET_LABEL[target]} — transfer per property", fontsize=11)
    handles, labels = _dedup_legend(flat)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=8,
        ncol=len(labels),
        columnspacing=0.6,
        handletextpad=0.2,
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
    )
    fig.tight_layout()
    plt.subplots_adjust(top=0.86 if nrows == 1 else 0.92, wspace=0.30, hspace=0.40)
    return fig


def plot_per_property(df: pd.DataFrame) -> None:
    pdf_path = FIGS / "transfer_r2_vs_source_per_property.pdf"
    with PdfPages(pdf_path) as pdf:
        for target, props, figsize, nrows, ncols in [
            ("cosmos-hsc", ["redshift", "mass", "sSFR"], (11, 3.5), 1, 3),
            ("cosmos-jwst", ["redshift", "mass", "sSFR"], (11, 3.5), 1, 3),
            ("gz10", ["redshift"], (4.5, 3.5), 1, 1),
            ("galaxies", GALAXIES_PROPERTIES, (13, 7.0), 3, 5),
        ]:
            fig = _per_property_page(df, target, props, figsize=figsize, ncols=ncols, nrows=nrows)
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {pdf_path}")


def plot_f1_panel(df: pd.DataFrame) -> None:
    panel = cast(
        pd.DataFrame,
        df[
            (df["target"] == "gz10")
            & (df["property"] == "gz10_label")
            & (df["kind"] == "classification")
        ],
    )
    if panel.empty:
        print("warning: no gz10 classification rows; skipping F1 panel")
        return
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))
    _plot_strip(ax, panel, "gz10", metric="f1")
    ax.set_title("GZ10 class (F1)", fontsize=11)
    ax.set_ylabel("F1 (macro)", fontsize=10)

    handles, labels = _dedup_legend([ax])
    fig.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=8,
        ncol=len(labels),
        columnspacing=0.6,
        handletextpad=0.2,
        bbox_to_anchor=(0.52, 1.20),
        frameon=False,
    )
    fig.tight_layout()
    out = FIGS / "transfer_f1_vs_source.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = load_all_transfer()
    plot_mean_grid(df)
    plot_comparison_strip(df)
    plot_per_property(df)
    plot_f1_panel(df)


if __name__ == "__main__":
    main()
