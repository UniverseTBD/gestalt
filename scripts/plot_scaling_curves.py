#!/usr/bin/env python3
"""Scaling curves: R² vs basket size k for `bench scaling`.

One clean line per `subset_kind ∈ {random, one_per_family}`. The main
two-panel figure uses the compact 396pt paper layout and the Figure 1
visual theme.
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
    "mass": r"$\log M_\star$",
    "sSFR": r"sSFR",
}

BASKET_STYLE = {
    "basket_mcca_whitened": {"label": "Gestalt", "color": "#ff7f0e"},
}
FIG_WIDTH_IN = 396 / 72
FIG_HEIGHT_IN = 2.5
TITLE_SIZE = 9
TICK_SIZE = 7.5
LABEL_SIZE = 10
LEGEND_SIZE = 8

SUBSET_STYLE = {
    "random": {"label": "random", "marker": "o", "linestyle": "-"},
    "one_per_family": {"label": "one per family", "marker": "^", "linestyle": "--"},
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
        per_draw = sub[sub["k"] == k].groupby(["subset_id", "seed"])["r2"].mean().to_numpy()
        per_draw = per_draw[np.isfinite(per_draw)]
        if per_draw.size == 0:
            continue
        means[i] = per_draw.mean()
    return np.asarray(ks, dtype=float), means


def _plot_panel(
    ax,
    df: pd.DataFrame,
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
            ks,
            means,
            color=color,
            marker=style["marker"],
            linestyle=style["linestyle"],
            ms=4,
            lw=1.2,
            label=f"Gestalt, {style['label']}",
        )

    ks_all = sorted(df["k"].dropna().unique())
    if ks_all:
        ax.set_xscale("log")
        ax.set_xticks(ks_all)
        ax.set_xticklabels([str(int(k)) for k in ks_all])
        ax.minorticks_off()
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    _inward_ticks(ax)


def plot_mean(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=False)
    for ax, modality in zip(axes, MODALITIES):
        _plot_panel(ax, df, modality, PROPERTIES, MODALITY_LABEL[modality])
        ax.set_xlabel("basket size $k$", fontsize=LABEL_SIZE)
        if ax is axes[0]:
            ax.set_ylabel(r"$R^2$", fontsize=LABEL_SIZE)

    handles, labels = _dedup_legend(axes)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        fontsize=LEGEND_SIZE,
        ncol=len(labels),
        columnspacing=0.6,
        handletextpad=0.2,
        bbox_to_anchor=(0.5, 0.18),
        frameon=False,
    )
    fig.subplots_adjust(left=0.128, right=0.974, bottom=0.48, top=0.87, wspace=0.18)
    out = FIGS / "scaling_curves.pdf"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")
    plt.close(fig)


def plot_per_property(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True)
    for r, modality in enumerate(MODALITIES):
        for c, prop in enumerate(PROPERTIES):
            ax = axes[r, c]
            _plot_panel(
                ax,
                df,
                modality,
                [prop],
                f"{MODALITY_LABEL[modality]}: {PROPERTY_LABEL[prop]}",
            )
            if c == 0:
                ax.set_ylabel(r"$R^2$", fontsize=LABEL_SIZE)
            if r == 1:
                ax.set_xlabel("basket size $k$", fontsize=LABEL_SIZE)
            ax.tick_params(labelsize=TICK_SIZE)

    handles, labels = _dedup_legend(axes.flat)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=LEGEND_SIZE,
        ncol=(len(labels) + 1) // 2,
        columnspacing=0.6,
        handletextpad=0.2,
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
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
    plot_mean(df)
    plot_per_property(df)


if __name__ == "__main__":
    main()
