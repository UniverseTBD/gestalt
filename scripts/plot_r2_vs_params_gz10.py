#!/usr/bin/env python3
"""GZ10 metric vs model parameter count, with basket averages overlaid.

Mirrors `plot_r2_vs_params_cosmos.py` (and ultimately the pu plotting
idiom) but for the `bench gz10` sweep: a 2-panel figure where panel A
shows R² for the redshift regression target and panel B shows macro-F1
for the gz10_label classification target. The two task types share one
basket of single-model points; only the y-axis metric differs.

Per the copy-adapt decision in the plan, this script is intentionally
self-contained — helpers are pasted from `plot_r2_vs_params_cosmos.py`
rather than imported, with `_aggregate_per_seed` / `collect_singles` /
`collect_baskets` generalized to take a `metric` column name so the
F1 panel can swap r2 → f1 with no other changes.
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
        "label": "Gestalt",
        "color": "#000000", "ls": "-",
    },
    "basket_concat_pca": {
        "label": r"Basket (concat$\to$PCA)",
        "color": "#555555", "ls": ":",
    },
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
    """Scatter one marker per (family, size); overlay log-linear trend + ρ."""
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


PANELS = [
    {
        "property": "redshift",
        "kind": "regression",
        "metric": "r2",
        "ylabel": r"$R^2$",
        "title": r"redshift ($R^2$)",
    },
    {
        "property": "gz10_label",
        "kind": "classification",
        "metric": "f1",
        "ylabel": "F1 (macro)",
        "title": "GZ10 class (F1)",
    },
]


def plot_two_panel(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=False)
    for ax, panel in zip(axes, PANELS):
        sub = df[
            (df["property"] == panel["property"])
            & (df["kind"] == panel["kind"])
        ]
        _plot_singles_scatter(ax, collect_singles(sub, metric=panel["metric"]))
        _plot_baskets(ax, collect_baskets(sub, metric=panel["metric"]))
        ax.set_title(panel["title"], fontsize=11)
        ax.set_xlabel("Parameters", fontsize=10)
        ax.set_ylabel(panel["ylabel"], fontsize=10)
        _inward_ticks(ax)

    handles, labels = _dedup_legend(axes)
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=8, ncol=(len(labels) + 1) // 2,
        columnspacing=0.6, handletextpad=0.2,
        bbox_to_anchor=(0.52, 1.12), frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.28)
    out = FIGS / "gz10_r2_vs_model_size.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA / "results_pca1024_gz10.parquet")
    plot_two_panel(df)


if __name__ == "__main__":
    main()
