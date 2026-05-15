"""Bazaar CLI: two user verbs (run / fit), a plot verb, and a bench verb.

- `bazaar run   <input>`            — embed + transform via shipped or supplied fit
- `bazaar fit   <input>`            — embed + fit a fresh BazaarFit
- `bazaar bench {cosmos,gz10,galaxies} ...` — the dataset benchmark sweeps
- `bazaar plot  --data <parquet>`   — render plots from a bench parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from bazaar import api
from bazaar.basket import BASKET, ensure_embeddings_downloaded
from bazaar.bench.cosmosweb import catalog_pass, run_cosmosweb
from bazaar.bench.plotting import render_plots

# ---------------------------------------------------------------------------
# Public verbs (run / fit)
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
        split=args.split,
        max_samples=args.max_samples,
        modality=args.modality,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        out=args.out,
    )
    return 0


# ---------------------------------------------------------------------------
# Bench verb: `bazaar bench {cosmos,gz10,galaxies}`
# ---------------------------------------------------------------------------

def _add_cosmos_bench_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--D", type=int, default=256, help="PCA components per model")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--test-size", type=int, default=5000)
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
                        "ablation that skips per-model dim reduction.")


def _add_dataset_bench_args(
    p: argparse.ArgumentParser,
    *,
    default_n_seeds: int,
    default_test_size: int,
    default_split: str,
    split_help: str = "Default split for this dataset.",
) -> None:
    """Shared argparse block for `bench gz10` and `bench galaxies`."""
    p.add_argument("--D", type=int, default=256, help="PCA components per model")
    p.add_argument("--n-seeds", type=int, default=default_n_seeds)
    p.add_argument("--test-size", type=int, default=default_test_size)
    p.add_argument("--split", default=default_split, help=split_help)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on rows ingested (default: full split)")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Per-model .npy embedding cache "
                        "(default: ~/.cache/bazaar/embeds)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--whiten", choices=["pca_zscore", "zscore"], default="pca_zscore",
                   help="Per-model whitener before MCCA. 'zscore' is a PCA ablation.")
    p.add_argument("--out", type=Path, required=True, help="Output parquet path")


def cmd_bench_cosmos(args: argparse.Namespace) -> int:
    args.emb_dir.mkdir(parents=True, exist_ok=True)
    telescopes = ["hsc", "jwst"]
    ensure_embeddings_downloaded(BASKET, telescopes, args.emb_dir,
                                 args.stream_script)
    params = catalog_pass(args.n_use)
    print({k: (v.shape, float(np.isfinite(v).mean()))
           for k, v in params.items()})

    all_rows: list[dict] = []
    for tele in telescopes:
        all_rows.extend(run_cosmosweb(
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
    fit_p.add_argument("--out", type=Path, required=True,
                       help="Output directory for the saved fit")
    fit_p.set_defaults(func=cmd_fit)

    bench_p = sub.add_parser("bench", help="benchmark sweeps")
    bench_sub = bench_p.add_subparsers(dest="dataset", required=True)

    cosmos_p = bench_sub.add_parser(
        "cosmos",
        help="COSMOS-Web HSC×JWST sweep (Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2)",
    )
    _add_cosmos_bench_args(cosmos_p)
    cosmos_p.set_defaults(func=cmd_bench_cosmos)

    gz10_p = bench_sub.add_parser(
        "gz10",
        help="UniverseTBD/mmu_gz10 sweep "
             "(classification on gz10_label + regression on redshift)",
    )
    _add_dataset_bench_args(
        gz10_p, default_n_seeds=5, default_test_size=2_500, default_split="train",
    )
    gz10_p.set_defaults(func=cmd_bench_gz10)

    galaxies_p = bench_sub.add_parser(
        "galaxies",
        help="Smith42/galaxies (v2.0) sweep "
             "— 13 paper-faithful regression targets from Sanjaripour+2026",
    )
    _add_dataset_bench_args(
        galaxies_p, default_n_seeds=5, default_test_size=5_000, default_split="test",
        split_help="Default is the 86k test split; pass 'train' for the full 8.5M.",
    )
    galaxies_p.set_defaults(func=cmd_bench_galaxies)

    plot_p = sub.add_parser("plot", help="Render plots + stats from a bench parquet")
    plot_p.add_argument("--data", type=Path, required=True)
    plot_p.add_argument("--suffix", type=str, default="")
    plot_p.add_argument("--figs-dir", type=Path, default=Path("figs"))
    plot_p.set_defaults(func=cmd_plot)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
