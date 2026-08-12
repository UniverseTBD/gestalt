"""Generate LaTeX tables for the three Bazaar benchmarks.

Reads parquet files from ``data/`` and writes one ``\\begin{table*}`` per
benchmark into ``figs/``. Wide tables are wrapped in ``\\resizebox`` so they
fit ``\\textwidth`` regardless of the active document class.

Run from the repo root:

    uv run python scripts/gen_latex_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGS = REPO / "assets" / "plots"


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


def clean_source(src: str) -> str:
    if src == "basket_mcca_whitened":
        return r"\textbf{Basket (MCCA)}"
    if src == "basket_mcca_mean":
        return r"\textbf{Basket (MCCA mean)}"
    if src == "basket_concat_pca":
        return r"\textbf{Basket (concat$\to$PCA)}"
    if src.startswith("single_"):
        name = src.removeprefix("single_").rsplit("_pca", 1)[0]
        return name.replace("_", r"\_")
    return src.replace("_", r"\_")


def is_basket(src: str) -> bool:
    return src.startswith("basket_")


BASKET_ORDER = {
    "basket_mcca_whitened": 0,
    "basket_mcca_mean":     0,
    "basket_concat_pca":    1,
}


def order_key(src: str) -> tuple[int, int, str]:
    return (0 if is_basket(src) else 1, BASKET_ORDER.get(src, 0), src)


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grp = (
        df.groupby(["source", "property"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["cell"] = list(zip(grp["mean"], grp["std"]))
    return grp.pivot(index="source", columns="property", values="cell")


def fmt_cell(mean: float, std: float, bold: bool) -> str:
    if np.isnan(mean):
        return "--"
    if bold:
        return f"$\\mathbf{{{mean:.3f} \\pm {std:.3f}}}$"
    return f"${mean:.3f} \\pm {std:.3f}$"


def render_table(
    wide: pd.DataFrame,
    properties: list[str],
    *,
    caption: str,
    label: str,
    source_order: list[str],
    metric_name: str,
    resize: bool = True,
    rotate: bool = False,
) -> str:
    best_per_prop: dict[str, str] = {}
    for prop in properties:
        means = {src: wide.loc[src, prop][0] for src in source_order if prop in wide.columns}
        best_per_prop[prop] = max(means, key=lambda s: means[s] if not np.isnan(means[s]) else -np.inf)

    n_cols = len(properties)
    align = "l" + "c" * n_cols
    header = " & ".join([""] + [PROPERTY_LABEL.get(p, p) for p in properties]) + r" \\"

    env = "sidewaystable*" if rotate else "table*"
    resize_target = r"\textheight" if rotate else r"\textwidth"

    lines: list[str] = []
    lines.append(rf"\begin{{{env}}}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    if resize:
        lines.append(rf"\resizebox{{{resize_target}}}{{!}}{{%")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    baskets = [s for s in source_order if is_basket(s)]
    singles = [s for s in source_order if not is_basket(s)]

    for block in (baskets, singles):
        for src in block:
            row = [clean_source(src)]
            for prop in properties:
                if prop in wide.columns and src in wide.index:
                    mean, std = wide.loc[src, prop]
                    row.append(fmt_cell(mean, std, bold=(best_per_prop[prop] == src)))
                else:
                    row.append("--")
            lines.append(" & ".join(row) + r" \\")
        if block is baskets:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append(r"}")
    lines.append(rf"\par\smallskip\textit{{Values are {metric_name} (mean $\pm$ std) across random probe seeds; best per column in bold.}}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines) + "\n"


def render_cosmos() -> str:
    df = pd.read_parquet(DATA / "cosmos_1024.parquet")
    df = df.copy()
    df["property"] = df["modality"].astype(str) + ":" + df["property"].astype(str)
    properties = ["hsc:redshift", "hsc:mass", "hsc:sSFR",
                  "jwst:redshift", "jwst:mass", "jwst:sSFR"]
    wide = aggregate(df, "r2")
    source_order = sorted(df["source"].unique(), key=order_key)

    PROPERTY_LABEL.update({
        "hsc:redshift":  r"$z$ (HSC)",
        "hsc:mass":      r"$\log M_\star$ (HSC)",
        "hsc:sSFR":      r"sSFR (HSC)",
        "jwst:redshift": r"$z$ (JWST)",
        "jwst:mass":     r"$\log M_\star$ (JWST)",
        "jwst:sSFR":     r"sSFR (JWST)",
    })

    return render_table(
        wide, properties,
        caption=(r"COSMOS-Web HSC$\times$JWST linear probe $R^2$ on 45\,000 galaxies "
                 r"(PCA-1024, $z$-scored; 10 probe seeds)."),
        label="tab:cosmosweb",
        source_order=source_order,
        metric_name=r"$R^2$",
        resize=True,
    )


def render_gz10() -> str:
    df = pd.read_parquet(DATA / "results_pca1024_gz10.parquet")
    cls = df[df["kind"] == "classification"].copy()
    reg = df[df["kind"] == "regression"].copy()
    cls["metric"] = cls["f1"]
    reg["metric"] = reg["r2"]
    cls["property"] = "gz10_label"
    reg["property"] = "redshift"
    long = pd.concat([cls[["source", "property", "metric"]],
                      reg[["source", "property", "metric"]]])

    grp = (
        long.groupby(["source", "property"])["metric"]
        .agg(["mean", "std"])
        .reset_index()
    )
    grp["cell"] = list(zip(grp["mean"], grp["std"]))
    wide = grp.pivot(index="source", columns="property", values="cell")

    properties = ["gz10_label", "redshift"]
    source_order = sorted(df["source"].unique(), key=order_key)

    return render_table(
        wide, properties,
        caption=(r"Galaxy Zoo 10 (UniverseTBD/mmu\_gz10) linear-probe scores: "
                 r"macro $F_1$ for the 10-way morphology classification and "
                 r"$R^2$ for photometric redshift (PCA-1024, $z$-scored; 5 probe seeds)."),
        label="tab:gz10",
        source_order=source_order,
        metric_name=r"macro $F_1$ / $R^2$",
        resize=False,
    )


def render_galaxies() -> str:
    df = pd.read_parquet(DATA / "results_pca1024_galaxies.parquet")
    wide = aggregate(df, "r2")
    properties = [
        "mag_abs_g", "mag_abs_z", "g_minus_r", "r_minus_z",
        "photo_z", "spec_z",
        "log_mstar", "mean_ssfr",
        "smooth", "disc", "artifact", "edge_on", "tight_spiral",
    ]
    properties = [p for p in properties if p in wide.columns]
    source_order = sorted(df["source"].unique(), key=order_key)

    return render_table(
        wide, properties,
        caption=(r"Smith42/galaxies (v2.0) linear-probe $R^2$ on 13 paper-faithful "
                 r"regression targets (PCA-1024, $z$-scored; 5 probe seeds)."),
        label="tab:galaxies",
        source_order=source_order,
        metric_name=r"$R^2$",
        resize=True,
        rotate=True,
    )


SCALING_MODALITY_LABEL = {"hsc": "HSC", "jwst": "JWST"}
SCALING_PROP_LABEL = {
    "redshift": r"$z_{\rm phot}$",
    "mass":     r"$\log M_\star$",
    "sSFR":     r"sSFR",
}
SCALING_KIND_LABEL = {"random": "random", "one_per_family": "one/family"}
SCALING_KS = [2, 4, 8, 16, 22]
SCALING_MODALITIES = ["hsc", "jwst"]
SCALING_PROPS = ["redshift", "mass", "sSFR"]
SCALING_KINDS = ["random", "one_per_family"]


def _scaling_means(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(modality, property, subset_kind, k) mean/std/n over subsets × seeds.

    The k = 22 `full` rows are folded into both `random` and `one_per_family`,
    matching the plotting convention.
    """
    full = df[df["subset_kind"] == "full"].copy()
    parts = [df[df["subset_kind"].isin(SCALING_KINDS)].copy()]
    for kind in SCALING_KINDS:
        f = full.copy()
        f["subset_kind"] = kind
        parts.append(f)
    long = pd.concat(parts, ignore_index=True)
    return (
        long.groupby(["modality", "property", "subset_kind", "k"])["r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def _scaling_law(k: np.ndarray, R_inf: float, A: float, alpha: float) -> np.ndarray:
    return R_inf - A * np.power(k, -alpha)


def _fit_scaling_law(ks: np.ndarray, means: np.ndarray) -> tuple[float, float, float, float]:
    p0 = [float(means.max()), max(1e-3, float(means.max() - means.min())), 1.0]
    popt, _ = curve_fit(
        _scaling_law, ks.astype(float), means,
        p0=p0, bounds=([0.0, 0.0, 0.05], [1.0, 5.0, 5.0]),
        maxfev=20000,
    )
    rmse = float(np.sqrt(np.mean((_scaling_law(ks.astype(float), *popt) - means) ** 2)))
    return float(popt[0]), float(popt[1]), float(popt[2]), rmse


def render_scaling_values() -> str:
    df = pd.read_parquet(DATA / "scaling.parquet")
    agg = _scaling_means(df)
    cell = {
        (r["modality"], r["property"], r["subset_kind"], int(r["k"])): (r["mean"], r["std"])
        for r in agg.to_dict("records")
    }

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption{COSMOS-Web basket-pruning $R^2$ vs basket size $k$ "
                 r"(MCCA-whitened fusion; mean $\pm$ std across subset draws and "
                 r"probe seeds). The $k{=}22$ column is the single full-basket "
                 r"value, folded into both subset policies; one-per-family is "
                 r"undefined for $k{>}8$ (only 8 model families).}")
    lines.append(r"\label{tab:scaling-values}")
    header_ks = " & ".join([rf"$k{{=}}{k}$" for k in SCALING_KS])
    lines.append(r"\begin{tabular}{lll" + "c" * len(SCALING_KS) + "}")
    lines.append(r"\toprule")
    lines.append(r"Modality & Property & Subset & " + header_ks + r" \\")
    lines.append(r"\midrule")

    first_in_modality = True
    for mod in SCALING_MODALITIES:
        if not first_in_modality:
            lines.append(r"\midrule")
        first_in_modality = False
        first_in_mod_block = True
        for prop in SCALING_PROPS:
            for ki, kind in enumerate(SCALING_KINDS):
                row = []
                row.append(SCALING_MODALITY_LABEL[mod] if first_in_mod_block else "")
                row.append(SCALING_PROP_LABEL[prop] if ki == 0 else "")
                row.append(SCALING_KIND_LABEL[kind])
                for k in SCALING_KS:
                    v = cell.get((mod, prop, kind, k))
                    if v is None or np.isnan(v[0]):
                        row.append("--")
                    else:
                        m, s = v
                        row.append(f"${m:.3f} \\pm {s:.3f}$")
                lines.append(" & ".join(row) + r" \\")
                first_in_mod_block = False

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def render_scaling_fits() -> str:
    df = pd.read_parquet(DATA / "scaling.parquet")
    agg = _scaling_means(df)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\caption{Fitted basket-size scaling law "
                 r"$R^2(k) = R^2_\infty - A\,k^{-\alpha}$ to the per-$k$ mean "
                 r"$R^2$ values in Table~\ref{tab:scaling-values}. RMSE is the "
                 r"residual on those means (units of $R^2$).}")
    lines.append(r"\label{tab:scaling-fits}")
    lines.append(r"\begin{tabular}{lllcccc}")
    lines.append(r"\toprule")
    lines.append(r"Modality & Property & Subset & $R^2_\infty$ & $A$ & $\alpha$ & RMSE \\")
    lines.append(r"\midrule")

    first_in_modality = True
    for mod in SCALING_MODALITIES:
        if not first_in_modality:
            lines.append(r"\midrule")
        first_in_modality = False
        first_in_mod_block = True
        for prop in SCALING_PROPS:
            for ki, kind in enumerate(SCALING_KINDS):
                sub = agg[(agg["modality"] == mod)
                          & (agg["property"] == prop)
                          & (agg["subset_kind"] == kind)].sort_values("k")
                ks = sub["k"].to_numpy()
                means = sub["mean"].to_numpy()
                if len(ks) < 3:
                    cells = ["--"] * 4
                else:
                    R_inf, A, alpha, rmse = _fit_scaling_law(ks, means)
                    cells = [f"${R_inf:.3f}$", f"${A:.3f}$",
                             f"${alpha:.2f}$", f"${rmse:.4f}$"]
                row = [
                    SCALING_MODALITY_LABEL[mod] if first_in_mod_block else "",
                    SCALING_PROP_LABEL[prop] if ki == 0 else "",
                    SCALING_KIND_LABEL[kind],
                    *cells,
                ]
                lines.append(" & ".join(row) + r" \\")
                first_in_mod_block = False

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


TRANSFER_TARGETS = [
    # (target slug, in-domain fit_source, rotate, caption, label_suffix)
    ("cosmos-hsc", "cosmos-hsc", False,
     (r"COSMOS-Web HSC cross-survey transfer: linear-probe $R^2$ on the "
      r"45\,000-galaxy HSC target when the basket fit is trained on each of "
      r"the four source corpora. $^{\dagger}$~marks the in-domain fit "
      r"(\textsc{cosmos-hsc} fit on \textsc{cosmos-hsc}); other rows quantify "
      r"the cost of fitting on a different survey."),
     "cosmos-hsc"),
    ("cosmos-jwst", "cosmos-jwst", False,
     (r"COSMOS-Web JWST cross-survey transfer: linear-probe $R^2$ on the "
      r"45\,000-galaxy JWST target when the basket fit is trained on each "
      r"of the four source corpora."),
     "cosmos-jwst"),
    ("gz10", "gz10", False,
     (r"Galaxy Zoo 10 cross-survey transfer: macro $F_1$ for the 10-way "
      r"morphology classification and $R^2$ for photometric redshift on "
      r"the GZ10 target when the basket fit is trained on each of the four "
      r"source corpora."),
     "gz10"),
    ("galaxies", "galaxies", True,
     (r"Smith42/galaxies cross-survey transfer: linear-probe $R^2$ on the "
      r"13 paper-faithful regression targets when the basket fit is trained "
      r"on each of the four source corpora."),
     "galaxies"),
]


def _transfer_metric(df: pd.DataFrame) -> pd.Series:
    return np.where(df["kind"] == "classification", df["f1"], df["r2"])


def render_transfer(
    target: str,
    in_domain: str,
    *,
    caption: str,
    label_suffix: str,
    rotate: bool,
) -> str:
    df = pd.read_parquet(DATA / f"transfer_{target}.parquet").copy()
    df["metric"] = _transfer_metric(df)

    properties = list(df["property"].drop_duplicates())
    fit_sources_all = sorted(df["fit_source"].unique())
    fit_sources = [in_domain] + [s for s in fit_sources_all if s != in_domain]
    sources = sorted(df["source"].unique(), key=order_key)

    grp = (df.groupby(["source", "fit_source", "property"])["metric"]
             .agg(["mean", "std"]).reset_index())
    cell = {(r["source"], r["fit_source"], r["property"]): (r["mean"], r["std"])
            for r in grp.to_dict("records")}

    best: dict[str, tuple[str, str] | None] = {}
    for prop in properties:
        best_v = -np.inf
        best_key: tuple[str, str] | None = None
        for src in sources:
            for fs in fit_sources:
                v = cell.get((src, fs, prop))
                if v is None:
                    continue
                m = v[0]
                if not np.isnan(m) and m > best_v:
                    best_v = m
                    best_key = (src, fs)
        best[prop] = best_key

    n_cols = len(properties)
    align = "ll" + "c" * n_cols
    env = "sidewaystable*" if rotate else "table*"
    resize_target = r"\textheight" if rotate else r"\textwidth"

    lines: list[str] = []
    lines.append(rf"\begin{{{env}}}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{tab:transfer-{label_suffix}}}")
    lines.append(rf"\resizebox{{{resize_target}}}{{!}}{{%")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")
    header = (" & ".join(["Source", "Fit on"]
                         + [PROPERTY_LABEL.get(p, p) for p in properties])
              + r" \\")
    lines.append(header)
    lines.append(r"\midrule")

    first_src = True
    for src in sources:
        if not first_src:
            lines.append(r"\midrule")
        first_src = False
        for i, fs in enumerate(fit_sources):
            src_label = clean_source(src) if i == 0 else ""
            fs_label = fs.replace("_", r"\_")
            if fs == in_domain:
                fs_label = fs_label + r"$^{\dagger}$"
            row = [src_label, fs_label]
            for prop in properties:
                v = cell.get((src, fs, prop))
                if v is None or np.isnan(v[0]):
                    row.append("--")
                else:
                    m, s = v
                    row.append(fmt_cell(m, s, bold=(best[prop] == (src, fs))))
            lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\par\smallskip\textit{Mean $\pm$ std across probe seeds; "
                 r"best per column in bold. $^{\dagger}$~marks the in-domain fit.}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines) + "\n"


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = {
        "cosmosweb_table.tex":      render_cosmos(),
        "gz10_table.tex":           render_gz10(),
        "galaxies_table.tex":       render_galaxies(),
        "scaling_values_table.tex": render_scaling_values(),
        "scaling_fits_table.tex":   render_scaling_fits(),
    }
    for target, in_domain, rotate, caption, label_suffix in TRANSFER_TARGETS:
        out[f"transfer_{target}_table.tex"] = render_transfer(
            target, in_domain,
            caption=caption, label_suffix=label_suffix, rotate=rotate,
        )
    for name, body in out.items():
        path = FIGS / name
        path.write_text(body)
        print(f"Wrote {path}  ({len(body)} bytes)")


if __name__ == "__main__":
    main()
