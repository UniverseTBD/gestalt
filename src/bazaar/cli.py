"""`bazaar run` and `bazaar plot` entrypoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from bazaar.basket import BASKET, ensure_embeddings_downloaded
from bazaar.pipeline import catalog_pass, run_modality
from bazaar.plotting import render_plots


def _add_run_args(p: argparse.ArgumentParser) -> None:
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


def cmd_run(args: argparse.Namespace) -> int:
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
        ))

    df = pl.DataFrame(all_rows).with_columns(pl.lit(args.D).alias("D"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"\n[bazaar] Wrote {len(df)} rows → {args.out}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    render_plots(args.data, args.figs_dir, args.suffix)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bazaar")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run basket sweep and write parquet")
    _add_run_args(run_p)
    run_p.set_defaults(func=cmd_run)

    plot_p = sub.add_parser("plot", help="Render plots + stats from parquet")
    plot_p.add_argument("--data", type=Path, required=True)
    plot_p.add_argument("--suffix", type=str, default="")
    plot_p.add_argument("--figs-dir", type=Path, default=Path("figs"))
    plot_p.set_defaults(func=cmd_plot)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
