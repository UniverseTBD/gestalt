"""Bazaar CLI: three user verbs (run / fit / load) + a hidden bench verb.

- `bazaar run  <input>`  — embed + transform via shipped or supplied fit
- `bazaar fit  <input>`  — embed + fit a fresh BazaarFit
- `bazaar load <fit_dir>`— inspect a saved fit
- `bazaar bench …`       — the legacy benchmark sweep (kept for reproducibility)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from bazaar import api
from bazaar.basket import BASKET, ensure_embeddings_downloaded
from bazaar.bench.pipeline import catalog_pass, run_modality
from bazaar.bench.plotting import render_plots
from bazaar.fit import BazaarFit

# ---------------------------------------------------------------------------
# Public verbs (run / fit / load)
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    api.run(
        args.input,
        fit=args.fit,
        split=args.split,
        max_samples=args.max_samples,
        modality=args.modality,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        out=args.out,
    )
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    api.fit(
        args.input,
        D=args.D,
        seed=args.seed,
        whiten_mode=args.whiten,
        split=args.split,
        max_samples=args.max_samples,
        modality=args.modality,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        out=args.out,
    )
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    fit = BazaarFit.load(args.fit_dir)
    meta = {
        "fit_dir": str(args.fit_dir),
        "D": fit.D,
        "seed": fit.seed,
        "whiten_mode": fit.whiten_mode,
        "basket_len": len(fit.basket),
        "basket": [list(t) for t in fit.basket],
        "V_shape": list(fit.mcca_V.shape) if fit.mcca_V is not None else None,
    }
    print(json.dumps(meta, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Hidden bench verb (legacy benchmark sweep)
# ---------------------------------------------------------------------------

def _add_bench_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--D", type=int, default=256, help="PCA components per model")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--test-size", type=int, default=5000)
    p.add_argument("--telescopes", nargs="+", default=["hsc", "jwst"])
    p.add_argument("--n-use", type=int, default=45_000)
    p.add_argument("--out", type=Path, required=True,
                   help="Output parquet path")
    p.add_argument("--emb-dir", type=Path,
                   default=Path("data/embeddings"),
                   help="Where the per-model .npy embedding cache lives")
    p.add_argument("--stream-script", type=Path,
                   default=Path(__file__).resolve().parents[2]
                   / "scripts" / "stream_embeddings_to_npy.py",
                   help="Path to the embedding downloader script")
    p.add_argument("--whiten", choices=["pca_zscore", "zscore"], default="pca_zscore",
                   help="Per-model whitener before MCCA. 'zscore' is a PCA "
                        "ablation; naive-mean and GPA are skipped in that mode.")


def cmd_bench(args: argparse.Namespace) -> int:
    args.emb_dir.mkdir(parents=True, exist_ok=True)
    ensure_embeddings_downloaded(BASKET, args.telescopes, args.emb_dir,
                                 args.stream_script)
    params = catalog_pass(args.n_use)
    print({k: (v.shape, float(np.isfinite(v).mean()))
           for k, v in params.items()})

    all_rows: list[dict] = []
    for tele in args.telescopes:
        all_rows.extend(run_modality(
            tele, params, BASKET,
            D=args.D, n_seeds=args.n_seeds,
            n_use=args.n_use, test_size=args.test_size,
            emb_dir=args.emb_dir,
            whiten_mode=args.whiten,
        ))

    df = (
        pl.DataFrame(all_rows)
        .with_columns(pl.lit(args.D).alias("D"))
        .with_columns(pl.lit(args.whiten).alias("whiten_mode"))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"\n[bazaar] Wrote {len(df)} rows → {args.out}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    render_plots(args.data, args.figs_dir, args.suffix)
    return 0


def cmd_bench_gz10(args: argparse.Namespace) -> int:
    from bazaar.bench.gz10 import run_gz10  # local: heavy imports (datasets, torch)

    rows = run_gz10(
        BASKET,
        D=args.D,
        n_seeds=args.n_seeds,
        test_size=args.test_size,
        split=args.split,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        whiten_mode=args.whiten,
    )
    df = (
        pl.DataFrame(rows)
        .with_columns(pl.lit(args.D).alias("D"))
        .with_columns(pl.lit(args.whiten).alias("whiten_mode"))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"\n[bazaar.gz10] Wrote {len(df)} rows → {args.out}")
    return 0


def cmd_bench_galaxies(args: argparse.Namespace) -> int:
    from bazaar.bench.galaxies import run_galaxies  # local: heavy imports

    rows = run_galaxies(
        BASKET,
        D=args.D,
        n_seeds=args.n_seeds,
        test_size=args.test_size,
        split=args.split,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        whiten_mode=args.whiten,
    )
    df = (
        pl.DataFrame(rows)
        .with_columns(pl.lit(args.D).alias("D"))
        .with_columns(pl.lit(args.whiten).alias("whiten_mode"))
        .with_columns(pl.lit(args.split).alias("split"))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"\n[bazaar.galaxies] Wrote {len(df)} rows → {args.out}")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def _add_run_fit_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("input",
                   help="HF dataset id or local path that `datasets.load_dataset` accepts")
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on the number of galaxies to ingest (default: all)")
    p.add_argument("--modality", default=None,
                   choices=["hsc", "jwst", "legacysurvey"],
                   help="Override band-set modality (default: inferred from image.band)")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Where per-model .npy embeddings are cached "
                        "(default: ~/.cache/bazaar/embeds)")
    p.add_argument("--batch-size", type=int, default=64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bazaar")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run",
                           help="Embed images and apply a saved fit; "
                                "writes a unified (N, D) .npy")
    _add_run_fit_shared(run_p)
    run_p.add_argument("--fit", default="default",
                       help="'default' downloads the shipped fit, or pass a directory")
    run_p.add_argument("--out", type=Path, default=Path("unified.npy"))
    run_p.set_defaults(func=cmd_run)

    fit_p = sub.add_parser("fit",
                           help="Embed images and fit a fresh BazaarFit to disk")
    _add_run_fit_shared(fit_p)
    fit_p.add_argument("--D", type=int, default=1024)
    fit_p.add_argument("--seed", type=int, default=0)
    fit_p.add_argument("--whiten", choices=["pca_zscore", "zscore"], default="zscore",
                       help="Per-model whitener (default: zscore = straight MCCA)")
    fit_p.add_argument("--out", type=Path, required=True,
                       help="Output directory for the saved fit")
    fit_p.set_defaults(func=cmd_fit)

    load_p = sub.add_parser("load",
                            help="Inspect a saved BazaarFit (prints meta + V shape)")
    load_p.add_argument("fit_dir", type=Path)
    load_p.set_defaults(func=cmd_load)

    # Hidden / legacy verbs (still discoverable via --help)
    bench_p = sub.add_parser("bench",
                             help="(legacy) the benchmark sweep that produced "
                                  "the published numbers")
    _add_bench_args(bench_p)
    bench_p.set_defaults(func=cmd_bench)

    plot_p = sub.add_parser("plot", help="(legacy) render plots + stats from bench parquet")
    plot_p.add_argument("--data", type=Path, required=True)
    plot_p.add_argument("--suffix", type=str, default="")
    plot_p.add_argument("--figs-dir", type=Path, default=Path("figs"))
    plot_p.set_defaults(func=cmd_plot)

    gz10_p = sub.add_parser(
        "bench-gz10",
        help="Benchmark sweep on UniverseTBD/mmu_gz10 "
             "(classification on gz10_label + regression on redshift)",
    )
    gz10_p.add_argument("--D", type=int, default=256, help="PCA components per model")
    gz10_p.add_argument("--n-seeds", type=int, default=5)
    gz10_p.add_argument("--test-size", type=int, default=2_500)
    gz10_p.add_argument("--split", default="train")
    gz10_p.add_argument("--max-samples", type=int, default=None,
                        help="Cap on rows ingested (default: full split)")
    gz10_p.add_argument("--cache-dir", type=Path, default=None,
                        help="Per-model .npy embedding cache "
                             "(default: ~/.cache/bazaar/embeds)")
    gz10_p.add_argument("--batch-size", type=int, default=64)
    gz10_p.add_argument("--whiten", choices=["pca_zscore", "zscore"], default="pca_zscore",
                        help="Per-model whitener before MCCA. 'zscore' is a PCA ablation.")
    gz10_p.add_argument("--out", type=Path, required=True,
                        help="Output parquet path")
    gz10_p.set_defaults(func=cmd_bench_gz10)

    galaxies_p = sub.add_parser(
        "bench-galaxies",
        help="Benchmark sweep on Smith42/galaxies (revision v2.0) "
             "— 13 paper-faithful regression targets from Sanjaripour+2026",
    )
    galaxies_p.add_argument("--D", type=int, default=256, help="PCA components per model")
    galaxies_p.add_argument("--n-seeds", type=int, default=5)
    galaxies_p.add_argument("--test-size", type=int, default=5_000)
    galaxies_p.add_argument("--split", default="test",
                            help="Default is the 86k test split; pass 'train' for "
                                 "the full 8.5M (very large embedding cache).")
    galaxies_p.add_argument("--max-samples", type=int, default=None,
                            help="Cap on rows ingested (default: full split)")
    galaxies_p.add_argument("--cache-dir", type=Path, default=None,
                            help="Per-model .npy embedding cache "
                                 "(default: ~/.cache/bazaar/embeds)")
    galaxies_p.add_argument("--batch-size", type=int, default=64)
    galaxies_p.add_argument("--whiten", choices=["pca_zscore", "zscore"],
                            default="pca_zscore",
                            help="Per-model whitener before MCCA. 'zscore' is a "
                                 "PCA ablation.")
    galaxies_p.add_argument("--out", type=Path, required=True,
                            help="Output parquet path")
    galaxies_p.set_defaults(func=cmd_bench_galaxies)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
