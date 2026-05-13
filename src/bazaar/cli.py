"""`bazaar run`, `plot`, `fit`, `transform`, and `embed` entrypoints."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import polars as pl

from bazaar.basket import (
    BASKET,
    ensure_default_fit_downloaded,
    ensure_embeddings_downloaded,
    load_embeddings,
)
from bazaar.embed import embed_via_pu
from bazaar.fit import BazaarFit
from bazaar.pipeline import catalog_pass, run_modality
from bazaar.plotting import render_plots

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "bazaar" / "embeds"
DEFAULT_FITS_DIR = Path.home() / ".cache" / "bazaar" / "fits"


def _resolve_emb_dir(args: argparse.Namespace) -> Path:
    """Return the directory containing per-model .npy files.

    Either pulls --emb-dir directly, or embeds via pu (--pu-path) into
    --cache-dir and returns that. Errors if neither is given.
    """
    if args.emb_dir is not None:
        return args.emb_dir
    pu_path = args.pu_path or (Path(os.environ["PU_PATH"]) if "PU_PATH" in os.environ else None)
    if pu_path is None:
        raise SystemExit(
            "must provide either --emb-dir (cached per-model embeddings) "
            "or --pu-path (run pu to produce them); PU_PATH env var also OK"
        )
    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
    embed_via_pu(
        modality=args.modality,
        pu_path=pu_path,
        cache_dir=cache_dir,
        basket=BASKET,
        test=getattr(args, "test", False),
    )
    return cache_dir


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
    p.add_argument("--whiten", choices=["pca_zscore", "zscore"], default="pca_zscore",
                   help="Per-model whitener before MCCA. 'zscore' is a PCA "
                        "ablation; naive-mean and GPA are skipped in that mode.")


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


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit a BazaarFit from per-model embeddings and save it."""
    emb_dir = _resolve_emb_dir(args)
    embeddings = load_embeddings(BASKET, args.modality, emb_dir, n_use=args.n_use)
    print(f"[bazaar] Fitting BazaarFit (D={args.D}, seed={args.seed}, "
          f"whiten={args.whiten}) on {len(BASKET)} models × {args.n_use} rows "
          f"({args.modality})...")
    fit = BazaarFit.fit(
        embeddings, basket=BASKET, D=args.D, seed=args.seed,
        whiten_mode=args.whiten,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    fit.save(args.out)
    print(f"[bazaar] Wrote fit → {args.out}/  ({len(fit.basket)} models, "
          f"V {fit.mcca_V.shape}, whiten={fit.whiten_mode})")
    return 0


def cmd_transform(args: argparse.Namespace) -> int:
    """Apply a saved BazaarFit to per-model embeddings."""
    if str(args.fit) == "default":
        fit_dir = ensure_default_fit_downloaded(args.modality, DEFAULT_FITS_DIR)
    else:
        fit_dir = args.fit
    fit = BazaarFit.load(fit_dir)
    emb_dir = _resolve_emb_dir(args)
    embeddings = load_embeddings(fit.basket, args.modality, emb_dir, n_use=args.n_use)
    print(f"[bazaar] Applying fit ({args.fit}) to {len(fit.basket)} models × "
          f"{args.n_use} rows ({args.modality})...")
    S = fit.transform(embeddings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, S)
    print(f"[bazaar] Wrote unified embedding {S.shape} → {args.out}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    """Pre-warm the per-model embedding cache by shelling out to pu."""
    pu_path = args.pu_path or Path(os.environ.get("PU_PATH", ""))
    if not pu_path or not pu_path.exists():
        raise SystemExit("--pu-path (or PU_PATH env var) must point to a pu checkout")
    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
    embed_via_pu(
        modality=args.modality,
        pu_path=pu_path,
        cache_dir=cache_dir,
        basket=BASKET,
        test=args.test,
    )
    print(f"[bazaar] Embeddings cached under {cache_dir}")
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

    def _add_source_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--emb-dir", type=Path, default=None,
                       help="Directory of per-model .npy embeddings (skips pu run)")
        p.add_argument("--pu-path", type=Path, default=None,
                       help="Path to a pu checkout; runs pu to produce embeddings "
                            "(falls back to env $PU_PATH)")
        p.add_argument("--cache-dir", type=Path, default=None,
                       help=f"Where pu embeddings are cached (default {DEFAULT_CACHE_DIR})")
        p.add_argument("--test", action="store_true",
                       help="Pass --test to pu run (small subset for smoke testing)")

    fit_p = sub.add_parser("fit", help="Fit a BazaarFit and save it to a directory")
    _add_source_args(fit_p)
    fit_p.add_argument("--modality", choices=["hsc", "jwst", "legacysurvey"], required=True)
    fit_p.add_argument("--D", type=int, default=256)
    fit_p.add_argument("--seed", type=int, default=0)
    fit_p.add_argument("--n-use", type=int, default=45_000)
    fit_p.add_argument("--whiten", choices=["pca_zscore", "zscore"],
                       default="pca_zscore",
                       help="Per-model whitener. 'zscore' ablates PCA "
                            "(keeps native d_m per model, MCCA only).")
    fit_p.add_argument("--out", type=Path, required=True,
                       help="Output directory for the saved fit")
    fit_p.set_defaults(func=cmd_fit)

    tr_p = sub.add_parser("transform",
                          help="Apply a saved BazaarFit to per-model embeddings")
    _add_source_args(tr_p)
    tr_p.add_argument("--fit", type=Path, required=True,
                      help="Path to a saved BazaarFit directory, or the literal "
                           "'default' to download the shipped COSMOS-Web D=256 fit")
    tr_p.add_argument("--modality", choices=["hsc", "jwst", "legacysurvey"], required=True)
    tr_p.add_argument("--n-use", type=int, default=45_000)
    tr_p.add_argument("--out", type=Path, required=True,
                      help="Output .npy path for the unified (N, D) embedding")
    tr_p.set_defaults(func=cmd_transform)

    emb_p = sub.add_parser("embed",
                           help="Pre-warm per-model embedding cache via pu run")
    emb_p.add_argument("--modality", choices=["hsc", "jwst", "legacysurvey"], required=True)
    emb_p.add_argument("--pu-path", type=Path, default=None)
    emb_p.add_argument("--cache-dir", type=Path, default=None)
    emb_p.add_argument("--test", action="store_true")
    emb_p.set_defaults(func=cmd_embed)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
