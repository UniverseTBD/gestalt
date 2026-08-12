#!/usr/bin/env python3
"""Plot member-model cosine ranges and Gestalt probe geometry.

The combined 3×3 matrix puts HSC in the lower triangle and JWST in the
upper triangle. Each off-diagonal cell shows the signed min–max range across
22 single models and the corresponding Gestalt value.

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
from matplotlib.patches import Rectangle

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
PROPERTIES = ("redshift", "mass", "sSFR")
PROPERTY_LABEL = {
    "redshift": r"$z$",
    "mass": r"$\log M_\star$",
    "sSFR": "sSFR",
}

FIG_WIDTH_IN = 396 / 72
FIG_HEIGHT_IN = 2.5
TITLE_SIZE = 8.5
PROPERTY_SIZE = 7.5
VALUE_SIZE = 6.5
LEGEND_SIZE = 7.5
RANGE_COLOR = "#7f7f7f"
GESTALT_COLOR = "#ff7f0e"
ZERO_COLOR = "#b0b0b0"
CELL_HALF_WIDTH = 0.40


def _inward_ticks(ax) -> None:
    for which in ("major", "minor"):
        ax.tick_params(axis="x", direction="in", which=which)
        ax.tick_params(axis="y", direction="in", which=which)


def _pivot_cos(df: pd.DataFrame, modality: str, source: str) -> np.ndarray:
    sub = df[(df["modality"] == modality) & (df["source"] == source)]
    if len(sub) != len(PROPERTIES) ** 2:
        raise ValueError(f"expected 9 cosine cells for {modality}/{source}, found {len(sub)}")

    index = {prop: i for i, prop in enumerate(PROPERTIES)}
    matrix = np.full((len(PROPERTIES), len(PROPERTIES)), np.nan, dtype=np.float64)
    for row in sub.itertuples(index=False):
        matrix[index[row.prop_i], index[row.prop_j]] = row.cos
    if not np.isfinite(matrix).all():
        raise ValueError(f"incomplete cosine matrix for {modality}/{source}")
    return matrix


def _member_ranges(df: pd.DataFrame, modality: str) -> tuple[np.ndarray, np.ndarray]:
    singles = sorted(source for source in df["source"].unique() if source.startswith("single_"))
    if len(singles) != 22:
        raise ValueError(f"expected 22 single-model sources, found {len(singles)}")
    stack = np.stack([_pivot_cos(df, modality, source) for source in singles])
    return np.min(stack, axis=0), np.max(stack, axis=0)


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


def _cell_x(column: int, value: float) -> float:
    return column + CELL_HALF_WIDTH * value


def _plot_range_cell(ax, row: int, column: int, low: float, high: float, gestalt: float) -> None:
    """Draw one local [-1, 1] range axis inside a matrix cell."""
    left = _cell_x(column, low)
    right = _cell_x(column, high)
    zero = _cell_x(column, 0.0)
    point = _cell_x(column, gestalt)

    ax.plot([zero, zero], [row - 0.13, row + 0.13], color=ZERO_COLOR, lw=0.6, zorder=1)
    ax.plot(
        [left, right],
        [row, row],
        color=RANGE_COLOR,
        lw=2.0,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot([left, left], [row - 0.07, row + 0.07], color=RANGE_COLOR, lw=1.0, zorder=2)
    ax.plot([right, right], [row - 0.07, row + 0.07], color=RANGE_COLOR, lw=1.0, zorder=2)
    ax.scatter(
        [point],
        [row],
        color=GESTALT_COLOR,
        edgecolor="white",
        linewidth=0.5,
        s=24,
        zorder=3,
    )

    # Place the value just below the marker without crowding neighboring cells.
    text_row = row + 0.19
    ax.text(
        point,
        text_row,
        f"{gestalt:+.2f}",
        ha="center",
        va="center",
        fontsize=VALUE_SIZE,
        color="#222222",
        zorder=4,
    )


def _plot_matrix(ax, df: pd.DataFrame, modality: str) -> None:
    low, high = _member_ranges(df, modality)
    gestalt = _pivot_cos(df, modality, "basket_mcca_whitened")

    for diagonal in range(len(PROPERTIES)):
        ax.add_patch(
            Rectangle(
                (diagonal - 0.5, diagonal - 0.5),
                1,
                1,
                facecolor="#f1f1f1",
                edgecolor="none",
                zorder=0,
            )
        )

    for row in range(len(PROPERTIES)):
        for column in range(len(PROPERTIES)):
            if row != column:
                _plot_range_cell(ax, row, column, low[row, column], high[row, column], gestalt[row, column])

    labels = [PROPERTY_LABEL[prop] for prop in PROPERTIES]
    ax.set_xlim(-0.5, len(PROPERTIES) - 0.5)
    ax.set_ylim(len(PROPERTIES) - 0.5, -0.5)
    ax.set_xticks(range(len(PROPERTIES)), labels=labels)
    ax.set_yticks(range(len(PROPERTIES)), labels=labels)
    ax.xaxis.tick_top()
    ax.set_title(modality.upper(), fontsize=TITLE_SIZE, pad=3)
    ax.tick_params(axis="both", which="major", labelsize=PROPERTY_SIZE, pad=2, length=3)
    ax.tick_params(axis="both", which="minor", length=0)
    _inward_ticks(ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    parquet_path, suffix = _resolve_parquet()
    df = pd.read_parquet(parquet_path)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    for ax, modality in zip(axes, MODALITIES):
        _plot_matrix(ax, df, modality)

    handles = [
        Line2D([0], [0], color=RANGE_COLOR, lw=2.0, label="22 member range"),
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
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.86, wspace=0.28)

    out = FIGS / f"probes_confusion{suffix}.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
