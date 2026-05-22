#!/usr/bin/env python3
"""Transfer R² (or F1) vs `fit_source` for `bench transfer`.

Categorical-X jittered strip-scatter, one marker per seed × source, one
panel per `target`. The reader's eye should pick up two patterns:

  1. On-domain (`fit_source == target`) sits at the top of each panel —
     drawn explicitly as a horizontal reference line at the MCCA mean.
  2. Cross-survey (cosmos ↔ legacysurvey) transfer is noticeably worse
     than intra-survey (cosmos-hsc ↔ cosmos-jwst); the Δ annotation in
     the bottom-right of each panel quantifies on-domain minus
     off-domain MCCA R².

Styling matches the cosmos figure family: orange MCCA / cyan concat→PCA,
top-of-figure legend, inward ticks, grid at alpha 0.25, figsize
parallels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGS = REPO / "figs"

FIT_SOURCES = ["cosmos-hsc", "cosmos-jwst", "gz10", "galaxies"]
TARGETS = ["cosmos-hsc", "cosmos-jwst", "gz10", "galaxies"]
TARGET_LABEL = {
    "cosmos-hsc":  "COSMOS-Web (HSC)",
    "cosmos-jwst": "COSMOS-Web (JWST)",
    "gz10":        "GZ10",
    "galaxies":    "Smith42/galaxies",
}

SOURCE_STYLE = {
    "basket_mcca_whitened": {
        "label": "Basket (MCCA, whitened)",
        "color": "#ff7f0e", "marker": "o", "offset": -0.20,
    },
    "basket_concat_pca": {
        "label": r"Basket (concat$\to$PCA)",
        "color": "#17becf", "marker": "s", "offset": 0.00,
    },
    "single_astropt_850M_pca_zscore": {
        "label": "AstroPT 850M (single)",
        "color": "#e377c2", "marker": "D", "offset": 0.20,
    },
}
SOURCE_ORDER = list(SOURCE_STYLE.keys())

# Galaxies per-property page (matches plot_r2_vs_params_galaxies.py).
GALAXIES_PROPERTIES = [
    "artifact", "disc", "edge_on", "g_minus_r", "log_mstar",
    "mag_abs_g", "mag_abs_z", "mean_ssfr", "photo_z", "r_minus_z",
    "smooth", "spec_z", "tight_spiral",
]
PROPERTY_LABEL = {
    "redshift":     r"$z_{\rm phot}$",
    "mass":         r"$\log M_\star$",
    "sSFR":         r"sSFR",
    "gz10_label":   r"GZ10 class",
    "photo_z":      r"$z_{\rm phot}$",
    "spec_z":       r"$z_{\rm spec}$",
    "mag_abs_g":    r"$M_g$",
    "mag_abs_z":    r"$M_z$",
    "g_minus_r":    r"$g{-}r$",
    "r_minus_z":    r"$r{-}z$",
    "log_mstar":    r"$\log M_\star$",
    "mean_ssfr":    r"sSFR",
    "smooth":       r"smooth",
    "disc":         r"disc",
    "artifact":     r"artifact",
    "edge_on":      r"edge-on",
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
        rows = panel[panel["source"] == src]
        if rows.empty:
            continue
        for i, fs in enumerate(FIT_SOURCES):
            vals = rows[rows["fit_source"] == fs][metric].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            x_center = i + style["offset"]
            jitter = rng.uniform(-width, width, size=vals.size)
            ax.scatter(
                x_center + jitter, vals,
                color=style["color"], marker=style["marker"],
                s=22, alpha=0.65, edgecolors="black", linewidths=0.3,
                label=style["label"] if i == 0 else None,
            )
            # Mean tick at the per-(src, fs) cluster center.
            ax.plot(
                [x_center - width, x_center + width],
                [vals.mean(), vals.mean()],
                color=style["color"], lw=1.6, solid_capstyle="butt",
            )

    # On-domain reference: MCCA mean at fit_source == target.
    on_dom = panel[
        (panel["source"] == "basket_mcca_whitened")
        & (panel["fit_source"] == target)
    ][metric].to_numpy()
    on_dom = on_dom[np.isfinite(on_dom)]
    if on_dom.size > 0:
        ax.axhline(
            on_dom.mean(),
            color=SOURCE_STYLE["basket_mcca_whitened"]["color"],
            lw=1.0, ls=":", alpha=0.85,
            label="on-domain MCCA mean",
        )

    # Δ = on-domain − off-domain MCCA mean.
    off_dom = panel[
        (panel["source"] == "basket_mcca_whitened")
        & (panel["fit_source"] != target)
    ][metric].to_numpy()
    off_dom = off_dom[np.isfinite(off_dom)]
    if on_dom.size > 0 and off_dom.size > 0:
        delta = on_dom.mean() - off_dom.mean()
        ax.text(
            0.95, 0.02,
            f"Δ = {delta:+.3f}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        )

    ax.set_xticks(range(len(FIT_SOURCES)))
    tick_labels = []
    for fs in FIT_SOURCES:
        if fs == target:
            tick_labels.append(f"{fs}\n(on-dom.)")
        else:
            tick_labels.append(fs)
    ax.set_xticklabels(tick_labels, fontsize=8)
    # Bold the on-domain tick label.
    for tl, fs in zip(ax.get_xticklabels(), FIT_SOURCES):
        if fs == target:
            tl.set_fontweight("bold")
    _inward_ticks(ax)


def plot_mean_grid(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), sharex=False)
    flat = list(axes.flat)
    for ax, target in zip(flat, TARGETS):
        panel = df[(df["target"] == target) & (df["kind"] == "regression")]
        # Mean across each target's regression properties per row first.
        agg = (
            panel.groupby(["fit_source", "source", "seed"], as_index=False)
            ["r2"].mean()
        )
        _plot_strip(ax, agg, target)
        ax.set_title(TARGET_LABEL[target], fontsize=11)
        ax.set_ylabel(r"Mean $R^2$", fontsize=10)

    handles, labels = _dedup_legend(flat)
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=8, ncol=len(labels),
        columnspacing=0.6, handletextpad=0.2,
        bbox_to_anchor=(0.5, 1.04), frameon=False,
    )
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.22, hspace=0.45)
    out = FIGS / "transfer_r2_vs_source.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def _per_property_page(
    df: pd.DataFrame, target: str, properties: list[str],
    *, figsize: tuple[float, float], ncols: int, nrows: int,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    panel_df = df[(df["target"] == target) & (df["kind"] == "regression")]
    flat = list(axes.flat)
    for idx, prop in enumerate(properties):
        ax = flat[idx]
        sub = panel_df[panel_df["property"] == prop]
        _plot_strip(ax, sub, target)
        ax.set_title(PROPERTY_LABEL.get(prop, prop), fontsize=10)
        if idx % ncols == 0:
            ax.set_ylabel(r"$R^2$", fontsize=9)
        ax.tick_params(labelsize=7)
    for idx in range(len(properties), nrows * ncols):
        flat[idx].set_visible(False)

    fig.suptitle(f"{TARGET_LABEL[target]} — transfer per property",
                 fontsize=11)
    handles, labels = _dedup_legend(flat)
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=8, ncol=len(labels),
        columnspacing=0.6, handletextpad=0.2,
        bbox_to_anchor=(0.5, 0.99), frameon=False,
    )
    fig.tight_layout()
    plt.subplots_adjust(top=0.86 if nrows == 1 else 0.92,
                        wspace=0.30, hspace=0.40)
    return fig


def plot_per_property(df: pd.DataFrame) -> None:
    pdf_path = FIGS / "transfer_r2_vs_source_per_property.pdf"
    with PdfPages(pdf_path) as pdf:
        for target, props, figsize, nrows, ncols in [
            ("cosmos-hsc",  ["redshift", "mass", "sSFR"], (11, 3.5), 1, 3),
            ("cosmos-jwst", ["redshift", "mass", "sSFR"], (11, 3.5), 1, 3),
            ("gz10",        ["redshift"],                 (4.5, 3.5), 1, 1),
            ("galaxies",    GALAXIES_PROPERTIES,          (13, 7.0),  3, 5),
        ]:
            fig = _per_property_page(df, target, props,
                                     figsize=figsize, ncols=ncols, nrows=nrows)
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {pdf_path}")


def plot_f1_panel(df: pd.DataFrame) -> None:
    panel = df[
        (df["target"] == "gz10")
        & (df["property"] == "gz10_label")
        & (df["kind"] == "classification")
    ]
    if panel.empty:
        print("warning: no gz10 classification rows; skipping F1 panel")
        return
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))
    _plot_strip(ax, panel, "gz10", metric="f1")
    ax.set_title("GZ10 class (F1)", fontsize=11)
    ax.set_ylabel("F1 (macro)", fontsize=10)

    handles, labels = _dedup_legend([ax])
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=8, ncol=len(labels),
        columnspacing=0.6, handletextpad=0.2,
        bbox_to_anchor=(0.52, 1.20), frameon=False,
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
    plot_per_property(df)
    plot_f1_panel(df)


if __name__ == "__main__":
    main()
