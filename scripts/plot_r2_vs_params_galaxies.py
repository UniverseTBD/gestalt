#!/usr/bin/env python3
"""Smith42/galaxies R² vs model parameter count, with basket averages overlaid.

Two figures:

  - `galaxies_r2_vs_model_size.pdf`: a single-panel scatter where y is
    the per-seed R² averaged across all 13 paper-faithful regression
    targets. Mirrors the cosmos `plot_mean` style but single-panel since
    `bench galaxies` has only one modality (legacysurvey).
  - `galaxies_r2_vs_model_size_per_property.pdf`: 3 × 5 grid with one
    panel per property (13 used, 2 hidden) in parquet order so it is
    easy to cross-reference the galaxies LaTeX table.

Self-contained — helpers are pasted from `plot_r2_vs_params_cosmos.py`
rather than imported, per the copy-adapt decision.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGS = REPO / "assets" / "plots"

# Parameter counts copied verbatim from pu/scripts/plot_r2_vs_params.py.
PARAM_COUNTS = {
    "vit": {
        "base": 86_389_248, "large": 304_351_232, "huge": 632_404_480,
    },
    "vit-mae": {
        "base": 86_389_248, "large": 304_351_232, "huge": 632_404_480,
    },
    "clip": {
        "base": 86_192_640, "large": 303_971_328,
    },
    "convnext": {
        "nano": 15_623_800, "tiny": 28_635_496,
        "base": 88_717_800, "large": 197_956_840,
    },
    "ijepa": {
        "huge": 630_762_240, "giant": 1_011_368_576,
    },
    "vjepa": {
        "large": 325_971_328, "huge": 653_930_880, "giant": 1_034_555_264,
    },
    "astropt": {
        "015M": 15_000_000, "095M": 95_000_000, "850M": 850_000_000,
    },
    "llava_15": {
        "7b": 7_062_898_688, "13b": 13_015_864_320,
    },
}

FAMILY_STYLE = {
    "vit":       {"label": "ViT",       "color": "#1f77b4", "marker": "o"},
    "vit-mae":   {"label": "ViT-MAE",   "color": "#efcc00", "marker": "<"},
    "clip":      {"label": "CLIP",      "color": "#ff7f0e", "marker": "s"},
    "convnext":  {"label": "ConvNeXt",  "color": "#2ca02c", "marker": "^"},
    "ijepa":     {"label": "I-JEPA",    "color": "#9467bd", "marker": "v"},
    "vjepa":     {"label": "V-JEPA",    "color": "#8c564b", "marker": "P"},
    "astropt":   {"label": "AstroPT",   "color": "#e377c2", "marker": "*"},
    "llava_15":  {"label": "LLaVA 1.5", "color": "#7f7f7f", "marker": "X"},
}

BASKET_STYLE = {
    "basket_mcca_whitened": {
        "label": "Basket (MCCA, whitened)",
        "color": "#000000", "ls": "-",
    },
    "basket_concat_pca": {
        "label": r"Basket (concat$\to$PCA)",
        "color": "#555555", "ls": ":",
    },
}

# Trimmed from scripts/gen_latex_tables.py PROPERTY_LABEL, keyed by the
# 13 galaxies properties.
PROPERTY_LABEL = {
    "artifact":     r"artifact",
    "disc":         r"disc",
    "edge_on":      r"edge-on",
    "g_minus_r":    r"$g{-}r$",
    "log_mstar":    r"$\log M_\star$",
    "mag_abs_g":    r"$M_g$",
    "mag_abs_z":    r"$M_z$",
    "mean_ssfr":    r"sSFR",
    "photo_z":      r"$z_{\rm phot}$",
    "r_minus_z":    r"$r{-}z$",
    "smooth":       r"smooth",
    "spec_z":       r"$z_{\rm spec}$",
    "tight_spiral": r"tight spiral",
}

# Parquet order, used for the per-property grid layout (matches the
# table in figs/galaxies_table.tex).
PROPERTY_ORDER = [
    "artifact", "disc", "edge_on", "g_minus_r", "log_mstar",
    "mag_abs_g", "mag_abs_z", "mean_ssfr", "photo_z", "r_minus_z",
    "smooth", "spec_z", "tight_spiral",
]


def parse_single(source: str) -> tuple[str, str] | None:
    """Split `single_<family>_<size>_pca<D>` into (family, size)."""
    if not source.startswith("single_"):
        return None
    core = source.removeprefix("single_")
    core = re.sub(r"_pca\d+$", "", core)
    family, sep, size = core.rpartition("_")
    if not sep:
        return None
    return family, size


def _aggregate_per_seed(
    sub: pd.DataFrame, metric: str = "r2"
) -> tuple[float, float, int]:
    """Mean ± SE of per-seed `metric` (averaged over whatever properties are in `sub`)."""
    per_seed = sub.groupby("seed")[metric].mean().to_numpy()
    per_seed = per_seed[np.isfinite(per_seed)]
    n = len(per_seed)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(per_seed.mean())
    se = float(per_seed.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, se, n


def collect_singles(
    df: pd.DataFrame, metric: str = "r2"
) -> dict[tuple[str, str], tuple[float, float]]:
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for src in df["source"].unique():
        parsed = parse_single(src)
        if parsed is None:
            continue
        m, se, n = _aggregate_per_seed(df[df["source"] == src], metric=metric)
        if n:
            out[parsed] = (m, se)
    return out


def collect_baskets(
    df: pd.DataFrame, metric: str = "r2"
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for src in df["source"].unique():
        if not src.startswith("basket_"):
            continue
        m, se, n = _aggregate_per_seed(df[df["source"] == src], metric=metric)
        if n:
            out[src] = (m, se)
    return out


def _fit_log_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS of y on log10(x). Returns (slope, intercept)."""
    slope, intercept = np.polyfit(np.log10(x), y, 1)
    return float(slope), float(intercept)


def _plot_singles_scatter(
    ax,
    singles: dict[tuple[str, str], tuple[float, float]],
    *,
    markersize: float = 30,
    annotate: bool = True,
    fit_line: bool = True,
) -> dict | None:
    xs_all, ys_all = [], []
    for family, params_by_size in PARAM_COUNTS.items():
        xs, ys = [], []
        for size, n_params in params_by_size.items():
            if (family, size) in singles:
                m, _ = singles[(family, size)]
                xs.append(float(n_params))
                ys.append(m)
        if not xs:
            continue
        style = FAMILY_STYLE[family]
        ax.scatter(
            xs, ys,
            color=style["color"], marker=style["marker"],
            s=markersize, label=style["label"],
            edgecolors="black", linewidths=0.4,
        )
        xs_all.extend(xs)
        ys_all.extend(ys)

    ax.set_xscale("log")
    if not xs_all:
        return None

    x_arr = np.asarray(xs_all)
    y_arr = np.asarray(ys_all)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if finite.sum() < 3:
        return None

    if annotate:
        rho, p_rho = spearmanr(x_arr[finite], y_arr[finite])
        ax.text(
            0.95, 0.02,
            f"ρ = {rho:.3f}  (p = {p_rho:.1g})",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        )

    slope, intercept = _fit_log_linear(x_arr[finite], y_arr[finite])
    if fit_line:
        xlim = ax.get_xlim()
        xfit = np.geomspace(xlim[0], xlim[1], 200)
        ax.plot(xfit, slope * np.log10(xfit) + intercept,
                color="gray", lw=1.5, ls="--", zorder=0)
        ax.set_xlim(xlim)

    return {"slope": slope, "intercept": intercept}


def _plot_baskets(ax, baskets: dict[str, tuple[float, float]],
                  *, with_band: bool = True) -> None:
    for src, (m, se) in baskets.items():
        style = BASKET_STYLE.get(src, {"label": src, "color": "k", "ls": "-"})
        ax.axhline(
            m, color=style["color"], linestyle=style["ls"],
            linewidth=1.2, alpha=0.75, label=style["label"],
        )
        if with_band and se > 0:
            ax.axhspan(m - se, m + se, color=style["color"], alpha=0.08)


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


def plot_mean(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))
    _plot_singles_scatter(ax, collect_singles(df))
    _plot_baskets(ax, collect_baskets(df))
    ax.set_xlabel("Parameters", fontsize=10)
    ax.set_ylabel(r"Mean $R^2$", fontsize=10)
    _inward_ticks(ax)

    handles, labels = _dedup_legend([ax])
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=8, ncol=(len(labels) + 1) // 2,
        columnspacing=0.6, handletextpad=0.2,
        bbox_to_anchor=(0.52, 1.18), frameon=False,
    )

    fig.tight_layout()
    out = FIGS / "galaxies_r2_vs_model_size.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_per_property(df: pd.DataFrame) -> None:
    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 6.5), sharex=True)
    for idx, prop in enumerate(PROPERTY_ORDER):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sub = df[df["property"] == prop]
        _plot_singles_scatter(ax, collect_singles(sub), markersize=22)
        _plot_baskets(ax, collect_baskets(sub))
        ax.set_title(PROPERTY_LABEL[prop], fontsize=10)
        if c == 0:
            ax.set_ylabel(r"$R^2$", fontsize=9)
        if r == nrows - 1:
            ax.set_xlabel("Parameters", fontsize=9)
        ax.tick_params(labelsize=8)
        _inward_ticks(ax)

    for idx in range(len(PROPERTY_ORDER), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    handles, labels = _dedup_legend(axes.flat)
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=8, ncol=(len(labels) + 1) // 2,
        columnspacing=0.6, handletextpad=0.2,
        bbox_to_anchor=(0.5, 1.03), frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.28, hspace=0.32)
    out = FIGS / "galaxies_r2_vs_model_size_per_property.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA / "results_pca1024_galaxies.parquet")
    # Guard against future evaluation splits — only the `test` rows are
    # the held-out scores plotted today.
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    plot_mean(df)
    plot_per_property(df)


if __name__ == "__main__":
    main()
