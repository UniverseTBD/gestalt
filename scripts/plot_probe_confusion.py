#!/usr/bin/env python3
"""Plot member-model probe-cosine violins and Gestalt values.

The figure has side-by-side HSC and JWST panels. Each panel contains one
violin per property pair, built from the 22 single-model cosine values, with
the corresponding Gestalt value overlaid.

Reads ``data/probes_1024.parquet`` and writes
``assets/plots/probes_confusion.pdf``. A smoke parquet is supported as a
fallback and produces a ``_smoke`` suffixed output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "DejaVu Sans"],
    }
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGS = REPO / "assets" / "plots"

MODALITIES = ("hsc", "jwst")
PROPERTY_PAIRS = (
    ("redshift", "mass"),
    ("redshift", "sSFR"),
    ("mass", "sSFR"),
)
PAIR_LABELS = (
    r"$z$–$\log M_\star$",
    r"$z$–sSFR",
    r"$\log M_\star$–sSFR",
)

FIG_WIDTH_IN = 396 / 72
FIG_HEIGHT_IN = 2.5
TITLE_SIZE = 9
TICK_SIZE = 7.5
VALUE_SIZE = 6.5
LABEL_SIZE = 8.5
LEGEND_SIZE = 7.5
MEMBER_COLOR = "#d9d9d9"
MEMBER_EDGE = "#7f7f7f"
GESTALT_COLOR = "#ff7f0e"
ZERO_COLOR = "#b0b0b0"


def _inward_ticks(ax) -> None:
    for which in ("major", "minor"):
        ax.tick_params(axis="x", direction="in", which=which)
        ax.tick_params(axis="y", direction="in", which=which)


def _resolve_parquet() -> tuple[Path, str]:
    full = DATA / "probes_1024.parquet"
    if full.exists():
        return full, ""
    smoke = DATA / "probes_smoke.parquet"
    if smoke.exists():
        print(
            "warning: data/probes_1024.parquet not found; "
            "falling back to data/probes_smoke.parquet (D=256). "
            "Output filename will be suffixed _smoke."
        )
        return smoke, "_smoke"
    raise FileNotFoundError("neither data/probes_1024.parquet nor data/probes_smoke.parquet exists")


def _member_values(
    df: pd.DataFrame, modality: str, prop_i: str, prop_j: str
) -> np.ndarray:
    singles = sorted(source for source in df["source"].unique() if source.startswith("single_"))
    if len(singles) != 22:
        raise ValueError(f"expected 22 single-model sources, found {len(singles)}")

    values = df[
        (df["modality"] == modality)
        & (df["source"].isin(singles))
        & (df["prop_i"] == prop_i)
        & (df["prop_j"] == prop_j)
    ]["cos"].to_numpy(dtype=np.float64)
    if len(values) != 22 or not np.isfinite(values).all():
        raise ValueError(f"incomplete member values for {modality}/{prop_i}/{prop_j}")
    return values


def _gestalt_value(df: pd.DataFrame, modality: str, prop_i: str, prop_j: str) -> float:
    values = df[
        (df["modality"] == modality)
        & (df["source"] == "basket_mcca_whitened")
        & (df["prop_i"] == prop_i)
        & (df["prop_j"] == prop_j)
    ]["cos"].to_numpy(dtype=np.float64)
    if len(values) != 1 or not np.isfinite(values[0]):
        raise ValueError(f"missing Gestalt value for {modality}/{prop_i}/{prop_j}")
    return values[0].item()


def _plot_panel(ax, df: pd.DataFrame, modality: str) -> None:
    for position, ((prop_i, prop_j), label) in enumerate(zip(PROPERTY_PAIRS, PAIR_LABELS), start=1):
        members = _member_values(df, modality, prop_i, prop_j)
        violin = ax.violinplot(
            [members],
            positions=[position],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(MEMBER_COLOR)
            body.set_edgecolor(MEMBER_EDGE)
            body.set_linewidth(0.7)
            body.set_alpha(1.0)

        gestalt = _gestalt_value(df, modality, prop_i, prop_j)
        ax.scatter(
            position,
            gestalt,
            color=GESTALT_COLOR,
            edgecolor="white",
            linewidth=0.5,
            s=24,
            zorder=3,
        )
        ax.annotate(
            f"{gestalt:+.2f}",
            (position, gestalt),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=VALUE_SIZE,
            color="#222222",
            zorder=4,
        )

    ax.axhline(0.0, color=ZERO_COLOR, lw=0.6, zorder=0)
    ax.set_title(modality.upper(), fontsize=TITLE_SIZE, pad=3)
    ax.set_xlim(0.45, len(PROPERTY_PAIRS) + 0.55)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(range(1, len(PAIR_LABELS) + 1), labels=PAIR_LABELS)
    ax.tick_params(axis="x", labelsize=TICK_SIZE, pad=2, length=3)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, length=3)
    _inward_ticks(ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    parquet_path, suffix = _resolve_parquet()
    df = pd.read_parquet(parquet_path)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=True)
    for ax, modality in zip(axes, MODALITIES):
        _plot_panel(ax, df, modality)
    axes[0].set_ylabel("probe cosine", fontsize=LABEL_SIZE)

    handles = [
        Patch(facecolor=MEMBER_COLOR, edgecolor=MEMBER_EDGE, label="22 member models"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GESTALT_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5,
            label="Gestalt",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=LEGEND_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.27, top=0.86, wspace=0.18)

    out = FIGS / f"probes_confusion{suffix}.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
