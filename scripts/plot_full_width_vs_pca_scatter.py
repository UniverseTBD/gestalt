#!/usr/bin/env python3
"""Scatter of full-width vs PCA-1024 single-model probe R² (paper figure).

One point per (modality, model, property) cell — 132 cells, mean over
10 seeds on each axis. Reads the same joined parquets as
`scripts/gen_full_width_table.py`; writes
`assets/plots/full_width_vs_pca_scatter.pdf`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGS = REPO / "assets" / "plots"
OUT_PDF = FIGS / "full_width_vs_pca_scatter.pdf"

MODALITY_STYLE = {
    "hsc": {"label": "HSC", "color": "#1f77b4", "marker": "o"},
    "jwst": {"label": "JWST", "color": "#d62728", "marker": "s"},
}
PROPERTY_MARKER_LABEL = {"redshift": r"$z$", "mass": r"$\log M_\star$", "sSFR": "sSFR"}


def main() -> None:
    pca_df = pd.read_parquet(DATA / "cosmos_1024.parquet")
    pca_df = pca_df[pca_df.source.str.startswith("single_")].copy()
    pca_df["model"] = pca_df.source.str.removeprefix("single_").str.removesuffix("_pca1024")
    pca_df = pca_df[["modality", "model", "property", "seed", "r2"]].rename(
        columns={"r2": "pca_r2"}
    )
    full_df = pd.read_parquet(DATA / "full_width_per_seed.parquet").rename(
        columns={"r2": "full_r2"}
    )
    paired = pca_df.merge(full_df, on=["modality", "model", "property", "seed"], how="inner")
    cell = (
        paired.groupby(["modality", "model", "property"])[["pca_r2", "full_r2"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    lo = float(min(cell.pca_r2.min(), cell.full_r2.min())) - 0.02
    hi = float(max(cell.pca_r2.max(), cell.full_r2.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], color="#7f7f7f", lw=0.9, ls="--", zorder=1)

    for (mod, prop), grp in cell.groupby(["modality", "property"]):
        style = MODALITY_STYLE[mod]
        ax.scatter(
            grp.full_r2,
            grp.pca_r2,
            s=22,
            color=style["color"],
            marker=style["marker"],
            alpha=0.75,
            zorder=2,
            label=f"{style['label']}, {PROPERTY_MARKER_LABEL[prop]}",
        )

    n_pca = int((cell.pca_r2 > cell.full_r2).sum())
    n_full = int((cell.pca_r2 < cell.full_r2).sum())
    delta = cell.pca_r2 - cell.full_r2
    ax.text(
        0.03,
        0.97,
        f"PCA-1024 above line: {n_pca}/{len(cell)}\n"
        f"full-width above: {n_full}/{len(cell)}\n"
        f"median $\\Delta R^2$: {delta.median():+.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    ax.set_xlabel(r"full-width $R^2$")
    ax.set_ylabel(r"PCA-1024 $R^2$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=7, frameon=False)

    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
