#!/usr/bin/env python3
"""Plot member-model probe-cosine violins, Gestalt values, and label relations.

The figure has side-by-side HSC and JWST panels. Each panel contains one
violin per property pair, built from the 22 single-model cosine values, with
the corresponding Gestalt value overlaid. Black diamonds show the shared
COSMOS-Web Pearson correlations after invalid-value masking and the same
1st/99th-percentile clipping used by the probes. They are a qualitative
relation anchor: label correlations and probe-weight cosines are not the
same quantity.

Reads ``data/probes_1024.parquet`` and ``data/cosmos_labels.npz`` and writes
``assets/plots/probes_confusion.pdf``. A smoke parquet is supported as a
fallback and produces a ``_smoke`` suffixed output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports]
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
FIG_HEIGHT_IN = 1.8
TITLE_SIZE = 9
TICK_SIZE = 7.5
VALUE_SIZE = 6.5
LABEL_SIZE = 8.5
LEGEND_SIZE = 7.5
MEMBER_COLOR = "#1f77b4"
GESTALT_COLOR = "#ff7f0e"
CATALOG_COLOR = "#111111"
MARKER_OFFSET = 0.08


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


def _catalog_correlations(path: Path) -> dict[tuple[str, str], float]:
    """Return clipped, pairwise Pearson correlations from COSMOS-Web labels."""
    arrays: dict[str, np.ndarray] = {}
    with np.load(path) as labels:
        for prop in ("redshift", "mass", "sSFR"):
            if prop not in labels:
                raise ValueError(f"missing {prop!r} in {path}")
            values = np.asarray(labels[prop], dtype=np.float64).copy()
            invalid = ~np.isfinite(values)
            if prop == "redshift":
                invalid |= values <= -50
            values[invalid] = np.nan
            finite = np.isfinite(values)
            if finite.sum() < 3:
                raise ValueError(f"fewer than 3 valid {prop} labels in {path}")
            lo, hi = np.quantile(values[finite], [0.01, 0.99])
            values[finite] = np.clip(values[finite], lo, hi)
            arrays[prop] = values

    correlations: dict[tuple[str, str], float] = {}
    for prop_i, prop_j in PROPERTY_PAIRS:
        finite = np.isfinite(arrays[prop_i]) & np.isfinite(arrays[prop_j])
        if finite.sum() < 3:
            raise ValueError(f"fewer than 3 paired labels for {prop_i}/{prop_j}")
        try:
            value = float(np.corrcoef(arrays[prop_i][finite], arrays[prop_j][finite])[0, 1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"could not compute catalog correlation for {prop_i}/{prop_j}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"non-finite catalog correlation for {prop_i}/{prop_j}")
        correlations[(prop_i, prop_j)] = value
    return correlations


def _member_values(df: pd.DataFrame, modality: str, prop_i: str, prop_j: str) -> np.ndarray:
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


def _plot_panel(
    ax,
    df: pd.DataFrame,
    modality: str,
    catalog_correlations: dict[tuple[str, str], float],
) -> None:
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
            body.set_edgecolor("none")
            body.set_linewidth(0.7)
            body.set_alpha(1.0)

        catalog = catalog_correlations[(prop_i, prop_j)]
        ax.scatter(
            position - MARKER_OFFSET,
            catalog,
            color=CATALOG_COLOR,
            marker="D",
            edgecolor="white",
            linewidth=0.5,
            s=25,
            zorder=3,
        )

        gestalt = _gestalt_value(df, modality, prop_i, prop_j)
        ax.scatter(
            position + MARKER_OFFSET,
            gestalt,
            color=GESTALT_COLOR,
            edgecolor="white",
            linewidth=0.5,
            s=24,
            zorder=4,
        )
        below = (prop_i, prop_j) == ("redshift", "sSFR")
        ax.annotate(
            f"{gestalt:+.2f}",
            (position + MARKER_OFFSET, gestalt),
            xytext=(0, -4 if below else 4),
            textcoords="offset points",
            ha="center",
            va="top" if below else "bottom",
            fontsize=VALUE_SIZE,
            color="#222222",
            zorder=4,
        )

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
    catalog_correlations = _catalog_correlations(DATA / "cosmos_labels.npz")

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=True)
    for ax, modality in zip(axes, MODALITIES):
        _plot_panel(ax, df, modality, catalog_correlations)
    axes[0].set_ylabel("probe cosine", fontsize=LABEL_SIZE)

    handles = [
        Patch(facecolor=MEMBER_COLOR, edgecolor="none", label="22 member models"),
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
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=CATALOG_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5,
            label="COSMOS-Web labels",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=LEGEND_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.129, right=0.973, bottom=0.282, top=0.820, wspace=0.185)

    out = FIGS / f"probes_confusion{suffix}.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
