#!/usr/bin/env python3
"""Scaling curves: R² vs basket size k for `bench scaling`.

One clean line per `subset_kind ∈ {random, one_per_family}`, plus
best-single and median-single horizontal references. The main two-panel
figure uses the compact 396pt paper layout and the Figure 1 visual theme.

Best/median-single references are read from `data/cosmos_1024.parquet`
so the basket-vs-single line matches the canonical D=1024 sweep used by
the model-size scatter plots.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
PROPERTIES = ["redshift", "mass", "sSFR"]
MODALITY_LABEL = {"hsc": "HSC", "jwst": "JWST"}
PROPERTY_LABEL = {
    "redshift": r"$z_{\rm phot}$",
    "mass":     r"$\log M_\star$",
    "sSFR":     r"sSFR",
}

BASKET_STYLE = {
    "basket_mcca_whitened": {"label": "Gestalt", "color": "#ff7f0e"},
}
FIG_WIDTH_IN = 396 / 72
FIG_HEIGHT_IN = 2.5

SUBSET_STYLE = {
    "random":         {"label": "random", "marker": "o", "linestyle": "-"},
    "one_per_family": {"label": "one per family", "marker": "^",
                       "linestyle": "--"},
}

SUBSET_KINDS = ("random", "one_per_family")
SOURCE = "basket_mcca_whitened"


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


def _scaling_curve(
    df: pd.DataFrame, modality: str, props: list[str], subset_kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Mean R² (across (subset_id, seed, property)) at each k for one curve.

    The `subset_kind=full` row at k=22 is folded in so each curve
    terminates at the all-22 basket value.
    """
    sub = df[
        (df["modality"] == modality)
        & (df["property"].isin(props))
        & (df["source"] == SOURCE)
        & (df["subset_kind"].isin([subset_kind, "full"]))
    ]
    if sub.empty:
        return np.array([]), np.array([])
    ks = sorted(sub["k"].unique())
    means = np.full(len(ks), np.nan, dtype=np.float64)
    for i, k in enumerate(ks):
        # average over properties within each (subset_id, seed) first so
        # the mean reflects between-seed/draw variability, not between-prop.
        per_draw = (
            sub[sub["k"] == k]
            .groupby(["subset_id", "seed"])["r2"]
            .mean()
            .to_numpy()
        )
        per_draw = per_draw[np.isfinite(per_draw)]
        if per_draw.size == 0:
            continue
        means[i] = per_draw.mean()
    return np.asarray(ks, dtype=float), means


def _single_refs(
    cosmos_df: pd.DataFrame, modality: str, props: list[str]
) -> tuple[float, float]:
    """(best_single_mean, median_single_mean) averaged across `props`."""
    sub = cosmos_df[
        (cosmos_df["modality"] == modality)
        & (cosmos_df["property"].isin(props))
        & (cosmos_df["source"].str.startswith("single_"))
    ]
    if sub.empty:
        return float("nan"), float("nan")
    per_model = (
        sub.groupby("source")["r2"].mean().to_numpy()
    )
    per_model = per_model[np.isfinite(per_model)]
    if per_model.size == 0:
        return float("nan"), float("nan")
    return float(per_model.max()), float(np.median(per_model))


def _plot_panel(
    ax,
    df: pd.DataFrame,
    cosmos_df: pd.DataFrame,
    modality: str,
    props: list[str],
    title: str,
) -> None:
    color = BASKET_STYLE[SOURCE]["color"]
    for kind in SUBSET_KINDS:
        ks, means = _scaling_curve(df, modality, props, kind)
        if ks.size == 0:
            continue
        style = SUBSET_STYLE[kind]
        ax.plot(
            ks, means,
            color=color, marker=style["marker"], linestyle=style["linestyle"],
            ms=4, lw=1.2, label=f"Gestalt, {style['label']}",
        )

    best, median = _single_refs(cosmos_df, modality, props)
    if np.isfinite(best):
        ax.axhline(best, color="#2ca02c", lw=1.0, ls=":", label="best single")
    if np.isfinite(median):
        ax.axhline(median, color="#7f7f7f", lw=1.0, ls="--", label="median single")

    ks_all = sorted(df["k"].dropna().unique())
    if ks_all:
        ax.set_xscale("log")
        ax.set_xticks(ks_all)
        ax.set_xticklabels([str(int(k)) for k in ks_all])
        ax.minorticks_off()
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    _inward_ticks(ax)


def plot_mean(df: pd.DataFrame, cosmos_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=False)
    for ax, modality in zip(axes, MODALITIES):
        _plot_panel(ax, df, cosmos_df, modality, PROPERTIES,
                    MODALITY_LABEL[modality])
        ax.set_xlabel("basket size $k$", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel(r"Mean $R^2$", fontsize=9)

    handles, labels = _dedup_legend(axes)
    fig.legend(
        handles, labels,
        loc="lower center", fontsize=7.5, ncol=len(labels),
        columnspacing=0.6, handletextpad=0.2,
        bbox_to_anchor=(0.5, 0.02), frameon=False,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    plt.subplots_adjust(wspace=0.22)
    out = FIGS / "scaling_curves.pdf"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")
    plt.close(fig)


def plot_per_property(df: pd.DataFrame, cosmos_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True)
    for r, modality in enumerate(MODALITIES):
        for c, prop in enumerate(PROPERTIES):
            ax = axes[r, c]
            _plot_panel(ax, df, cosmos_df, modality, [prop],
                        f"{MODALITY_LABEL[modality]}: {PROPERTY_LABEL[prop]}")
            if c == 0:
                ax.set_ylabel(r"$R^2$", fontsize=9)
            if r == 1:
                ax.set_xlabel("basket size $k$", fontsize=9)
            ax.tick_params(labelsize=8)

    handles, labels = _dedup_legend(axes.flat)
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=8, ncol=(len(labels) + 1) // 2,
        columnspacing=0.6, handletextpad=0.2,
        bbox_to_anchor=(0.5, 1.04), frameon=False,
    )
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.22, hspace=0.28)
    out = FIGS / "scaling_curves_per_property.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA / "scaling.parquet")
    cosmos_df = pd.read_parquet(DATA / "cosmos_1024.parquet")
    plot_mean(df, cosmos_df)
    plot_per_property(df, cosmos_df)


if __name__ == "__main__":
    main()
