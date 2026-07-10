#!/usr/bin/env python3
"""3×3 cosine confusion heatmaps for `bench probes`.

3×3 RdBu_r heatmap with cell text, laid out as a tight
2 (modality) × 3 (source) grid sized to fit the cosmos figure family.

Columns:
  1. `basket_mcca_whitened` — the canonical Bazaar projection.
  2. `basket_avg` — element-wise mean of the basket members' coefficient
     vectors (then probed).
  3. Median single — element-wise median over the 22 `single_*` cosine
     matrices, summarising "what a typical single model looks like."

Falls back to `data/probes_smoke.parquet` (D=256) with a `_smoke`
filename suffix if `data/probes_1024.parquet` is not yet present.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGS = REPO / "figs"

MODALITIES = ["hsc", "jwst"]
MODALITY_LABEL = {"hsc": "HSC", "jwst": "JWST"}
PROPERTIES = ["redshift", "mass", "sSFR"]
PROPERTY_LABEL = {
    "redshift": r"$z$",
    "mass":     r"$\log M_\star$",
    "sSFR":     "sSFR",
}

COLUMNS = [
    ("basket_mcca_whitened", "MCCA basket"),
    ("basket_avg",            "Basket avg"),
    ("__median_single__",     "Median single"),
]


def _inward_ticks(ax) -> None:
    for which in ("major", "minor"):
        ax.tick_params(axis="x", direction="in", which=which)
        ax.tick_params(axis="y", direction="in", which=which)


def _pivot_cos(df: pd.DataFrame, modality: str, source: str) -> np.ndarray:
    sub = df[(df["modality"] == modality) & (df["source"] == source)]
    idx = {p: i for i, p in enumerate(PROPERTIES)}
    M = np.full((3, 3), np.nan, dtype=np.float32)
    for r in sub.itertuples(index=False):
        M[idx[r.prop_i], idx[r.prop_j]] = r.cos
    return M


def _median_single(df: pd.DataFrame, modality: str) -> np.ndarray:
    singles = sorted(
        s for s in df["source"].unique() if s.startswith("single_")
    )
    if not singles:
        raise ValueError(f"no single_* sources for modality={modality!r}")
    stack = np.stack(
        [_pivot_cos(df, modality, s) for s in singles], axis=0,
    )
    return np.nanmedian(stack, axis=0)


def _combine_triangles(M_hsc: np.ndarray, M_jwst: np.ndarray) -> np.ndarray:
    """Lower triangle = HSC, upper triangle = JWST, diagonal = 1.0.

    Both modalities are expected to have a self-cosine of +1 on the
    diagonal; we take JWST's diagonal but they agree to float-precision.
    """
    M = np.empty_like(M_hsc)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if i > j:
                M[i, j] = M_hsc[i, j]
            elif i < j:
                M[i, j] = M_jwst[i, j]
            else:
                M[i, j] = M_jwst[i, j]
    return M


def _plot_heatmap(ax, M: np.ndarray, *, vmax: float = 1.0):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    labels = [PROPERTY_LABEL[p] for p in PROPERTIES]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            v = M[i, j]
            if not np.isfinite(v):
                continue
            ax.text(
                j, i, f"{v:+.2f}",
                ha="center", va="center",
                color="black" if abs(v) < 0.55 else "white",
                fontsize=9,
            )
    # Diagonal separator + triangle labels (placed slightly off the
    # diagonal in each triangle).
    ax.plot([-0.5, 2.5], [-0.5, 2.5], color="white", lw=2.5, zorder=3)
    ax.text(1.55, 0.45, "JWST", color="white",
            ha="center", va="center", fontsize=10, fontweight="bold",
            fontstyle="italic", zorder=4)
    ax.text(0.45, 1.55, "HSC", color="white",
            ha="center", va="center", fontsize=10, fontweight="bold",
            fontstyle="italic", zorder=4)
    _inward_ticks(ax)
    return im


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
    raise FileNotFoundError(
        "neither data/probes_1024.parquet nor data/probes_smoke.parquet exists"
    )


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    parquet_path, suffix = _resolve_parquet()
    df = pd.read_parquet(parquet_path)

    fig, axes = plt.subplots(1, len(COLUMNS), figsize=(8.5, 3.2))
    last_im = None
    for c, (src_key, col_label) in enumerate(COLUMNS):
        ax = axes[c]
        if src_key == "__median_single__":
            M_hsc = _median_single(df, "hsc")
            M_jwst = _median_single(df, "jwst")
        else:
            M_hsc = _pivot_cos(df, "hsc", src_key)
            M_jwst = _pivot_cos(df, "jwst", src_key)
        M = _combine_triangles(M_hsc, M_jwst)
        last_im = _plot_heatmap(ax, M)
        ax.set_title(col_label, fontsize=11)

    fig.tight_layout()
    plt.subplots_adjust(right=0.90, wspace=0.30)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.66])
    fig.colorbar(
        last_im, cax=cbar_ax,
        label="cosine of probe coefficients",
    )

    out = FIGS / f"probes_confusion{suffix}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
