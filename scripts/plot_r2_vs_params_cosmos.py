#!/usr/bin/env python3
"""COSMOS-Web R² vs model parameter count, with basket averages overlaid.

Mirrors the scatter style of `pu/scripts/plot_r2_vs_params.py` (on the
`crossmodalintramodal` branch): one marker per (family, size), a gray
log-linear regression line through all single-model points, and a
Spearman ρ + p annotation. Adds horizontal bands for each basket-fusion
variant so the single-model scatter can be compared against the basket
in one frame — that overlay is the bazaar-specific extension.
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
FIGS = REPO / "figs"

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

MODALITIES = ["hsc", "jwst"]
PROPERTIES = ["redshift", "mass", "sSFR"]
MODALITY_LABEL = {"hsc": "HSC", "jwst": "JWST"}
PROPERTY_LABEL = {
    "redshift": r"$z_{\rm phot}$",
    "mass":     r"$\log M_\star$",
    "sSFR":     r"sSFR",
}


def parse_single(source: str) -> tuple[str, str] | None:
    """Split `single_<family>_<size>_pca<D>` into (family, size).

    `llava_15` is the only family whose name contains an underscore, so we
    rsplit once on `_` after stripping the prefix/suffix — that puts the
    underscore back into the family token.
    """
    if not source.startswith("single_"):
        return None
    core = source.removeprefix("single_")
    core = re.sub(r"_pca\d+$", "", core)
    family, sep, size = core.rpartition("_")
    if not sep:
        return None
    return family, size


def _aggregate_per_seed(sub: pd.DataFrame) -> tuple[float, float, int]:
    """Mean ± SE of per-seed R² (averaged over whatever properties are in `sub`)."""
    per_seed = sub.groupby("seed")["r2"].mean().to_numpy()
    n = len(per_seed)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(per_seed.mean())
    se = float(per_seed.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, se, n


def collect_singles(df: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for src in df["source"].unique():
        parsed = parse_single(src)
        if parsed is None:
            continue
        m, se, n = _aggregate_per_seed(df[df["source"] == src])
        if n:
            out[parsed] = (m, se)
    return out


def collect_baskets(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for src in df["source"].unique():
        if not src.startswith("basket_"):
            continue
        m, se, n = _aggregate_per_seed(df[df["source"] == src])
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
    """Scatter one marker per (family, size); overlay log-linear trend + ρ.

    Returns ``{"slope", "intercept"}`` so callers can solve the trend for the
    equivalent-N at each basket's R² using the same fit drawn here.
    """
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
    """One panel per modality; y = R² averaged across the 3 COSMOS properties."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=False)
    for ax, modality in zip(axes, MODALITIES):
        sub = df[df["modality"] == modality]
        _plot_singles_scatter(ax, collect_singles(sub))
        _plot_baskets(ax, collect_baskets(sub))
        ax.set_title(MODALITY_LABEL[modality], fontsize=11)
        ax.set_xlabel("Parameters", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(r"Mean $R^2$", fontsize=10)
        ax.grid(True, alpha=0.25)
        _inward_ticks(ax)

    handles, labels = _dedup_legend(axes)
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=8, ncol=(len(labels) + 1) // 2,
        columnspacing=0.6, handletextpad=0.2,
        bbox_to_anchor=(0.52, 1.12), frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.22)
    out = FIGS / "cosmos_r2_vs_model_size.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_per_property(df: pd.DataFrame) -> None:
    """2 (modality) × 3 (property) grid in the same scatter idiom."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True)
    for r, modality in enumerate(MODALITIES):
        for c, prop in enumerate(PROPERTIES):
            ax = axes[r, c]
            sub = df[(df["modality"] == modality) & (df["property"] == prop)]
            _plot_singles_scatter(ax, collect_singles(sub), markersize=22)
            _plot_baskets(ax, collect_baskets(sub))
            ax.set_title(f"{MODALITY_LABEL[modality]}: {PROPERTY_LABEL[prop]}",
                         fontsize=10)
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.set_ylabel(r"$R^2$", fontsize=9)
            if r == 1:
                ax.set_xlabel("Parameters", fontsize=9)
            ax.tick_params(labelsize=8)
            _inward_ticks(ax)

    handles, labels = _dedup_legend(axes.flat)
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=8, ncol=(len(labels) + 1) // 2,
        columnspacing=0.6, handletextpad=0.2,
        bbox_to_anchor=(0.5, 1.04), frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.22, hspace=0.28)
    out = FIGS / "cosmos_r2_vs_model_size_per_property.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA / "cosmos_1024.parquet")
    plot_mean(df)
    plot_per_property(df)


if __name__ == "__main__":
    main()
