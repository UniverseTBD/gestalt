"""Bazaar CLI: two user verbs (run / fit) and a bench verb.

- `bazaar run   <input>`            — embed + transform via shipped or supplied fit
- `bazaar fit   <input>`            — embed + fit a fresh BazaarFit
- `bazaar bench {cosmos,gz10,galaxies} ...` — the dataset benchmark sweeps
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from bazaar import api
from bazaar.basket import BASKET, ensure_embeddings_downloaded
from bazaar.bench.cosmosweb import catalog_pass, run_cosmosweb

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
                   default=Path("embeds"),
                   help="Where the per-model .npy embedding cache lives "
                        "(default: ./embeds)")
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
                        "(default: ./embeds)")
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


def _parse_ks(s: str) -> list[int]:
    """Parse a comma-separated `--ks` argument (e.g. '2,4,8,16,22')."""
    out = [int(tok) for tok in s.split(",") if tok.strip()]
    if not out:
        raise argparse.ArgumentTypeError("--ks must list at least one k")
    if any(k <= 0 for k in out):
        raise argparse.ArgumentTypeError("--ks values must be positive")
    return out


def cmd_bench_scaling(args: argparse.Namespace) -> int:
    from bazaar.bench.cosmosweb import catalog_pass  # local: heavy imports
    from bazaar.bench.scaling import run_scaling_cosmos

    args.emb_dir.mkdir(parents=True, exist_ok=True)
    telescopes = ["hsc", "jwst"]
    ensure_embeddings_downloaded(BASKET, telescopes, args.emb_dir,
                                 args.stream_script)
    params = catalog_pass(args.n_use)
    print({k: (v.shape, float(np.isfinite(v).mean()))
           for k, v in params.items()})

    all_rows: list[dict] = []
    for tele in telescopes:
        all_rows.extend(run_scaling_cosmos(
            tele, params, BASKET,
            D=args.D, ks=args.ks,
            n_random_per_k=args.n_random_per_k,
            n_one_per_family=args.n_one_per_family,
            n_seeds=args.n_seeds,
            n_use=args.n_use, test_size=args.test_size,
            emb_dir=args.emb_dir,
            whiten_mode=args.whiten,
            subset_seed=args.subset_seed,
        ))

    df = (
        pl.DataFrame(all_rows, infer_schema_length=None)
        .with_columns(pl.lit(args.D).alias("D"))
        .with_columns(pl.lit(args.whiten).alias("whiten_mode"))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"\n[bazaar.scaling] Wrote {len(df)} rows → {args.out}")
    return 0


def cmd_bench_dims(args: argparse.Namespace) -> int:
    from bazaar.bench.dimensions import run_dimensions_cosmos  # local: heavy imports

    args.emb_dir.mkdir(parents=True, exist_ok=True)
    telescopes = ("hsc", "jwst")
    ensure_embeddings_downloaded(BASKET, list(telescopes), args.emb_dir,
                                 args.stream_script)
    rows = run_dimensions_cosmos(
        D=args.D, n_use=args.n_use,
        emb_dir=args.emb_dir, telescopes=telescopes, seed=args.seed,
    )
    df = pl.DataFrame(rows, infer_schema_length=None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"\n[bazaar.dims] Wrote {len(df)} rows → {args.out}")
    return 0


def cmd_bench_probes(args: argparse.Namespace) -> int:
    from bazaar.bench.probes import run_probes_cosmos  # local: heavy imports

    args.emb_dir.mkdir(parents=True, exist_ok=True)
    telescopes = ("hsc", "jwst")
    ensure_embeddings_downloaded(BASKET, list(telescopes), args.emb_dir,
                                 args.stream_script)
    rows = run_probes_cosmos(
        D=args.D, n_use=args.n_use,
        emb_dir=args.emb_dir, telescopes=telescopes, seed=args.seed,
    )
    df = pl.DataFrame(rows, infer_schema_length=None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"\n[bazaar.probes] Wrote {len(df)} rows → {args.out}")
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


def _parse_sources(s: str) -> list[str]:
    """Parse `--sources cosmos-hsc,gz10,...` (special token 'all' expands)."""
    from bazaar.bench.transfer import CORPORA
    toks = [tok.strip() for tok in s.split(",") if tok.strip()]
    if not toks:
        raise argparse.ArgumentTypeError("--sources must list at least one corpus")
    if toks == ["all"]:
        return list(CORPORA)
    out: list[str] = []
    for tok in toks:
        if tok == "all":
            raise argparse.ArgumentTypeError("'all' cannot be mixed with explicit sources")
        if tok not in CORPORA:
            raise argparse.ArgumentTypeError(
                f"unknown source {tok!r}; expected from {CORPORA}"
            )
        out.append(tok)
    return out


def cmd_bench_transfer(args: argparse.Namespace) -> int:
    from bazaar.bench.transfer import CORPORA, run_transfer  # local: heavy imports

    if args.target not in CORPORA:
        raise SystemExit(f"unknown --target {args.target!r}; expected from {CORPORA}")

    args.emb_dir.mkdir(parents=True, exist_ok=True)
    # COSMOS embeddings live as cached .npy; download any missing before fitting.
    cosmos_telescopes = [
        m for m in ("hsc", "jwst")
        if f"cosmos-{m}" in args.sources or f"cosmos-{m}" == args.target
    ]
    if cosmos_telescopes:
        ensure_embeddings_downloaded(
            BASKET, cosmos_telescopes, args.emb_dir, args.stream_script,
        )

    rows = run_transfer(
        args.target, args.sources,
        n_fit=args.n_fit, D=args.D, n_seeds=args.n_seeds,
        test_size=args.test_size,
        emb_dir=args.emb_dir, cache_dir=args.cache_dir,
        target_split=args.target_split,
        target_max_samples=args.target_max_samples,
        target_n_use=args.target_n_use,
        source_split=args.source_split,
        source_max_samples=args.source_max_samples,
        source_n_use=args.source_n_use,
        batch_size=args.batch_size, seed=args.seed,
    )
    df = pl.DataFrame(rows, infer_schema_length=None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"\n[bazaar.transfer] Wrote {len(df)} rows → {args.out}")
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
                        "(default: ./embeds)")
    p.add_argument("--batch-size", type=int, default=64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bazaar")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run",
                           help="Embed images and apply a saved fit; "
                                "writes a unified (N, D) .npy")
    _add_run_fit_shared(run_p)
    run_p.add_argument("--fit", default="default",
                       help="'default' (shipped fit), a local directory, "
                            "or a HuggingFace repo id (e.g. org/repo)")
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

    scaling_p = bench_sub.add_parser(
        "scaling",
        help="Basket pruning curves on COSMOS-Web — sweep R² vs basket size k "
             "under random and one-per-family subset selection.",
    )
    _add_cosmos_bench_args(scaling_p)
    scaling_p.add_argument(
        "--ks", type=_parse_ks, default=_parse_ks("2,4,8,16,22"),
        help="Comma-separated basket sizes (default: 2,4,8,16,22). "
             "k == len(basket) is always evaluated as the single 'full' subset.",
    )
    scaling_p.add_argument(
        "--n-random-per-k", type=int, default=8,
        help="Random subset draws per k (default: 8). "
             "Ignored at k == len(basket).",
    )
    scaling_p.add_argument(
        "--n-one-per-family", type=int, default=8,
        help="One-per-family subset draws per k (default: 8). "
             "Skipped when k > n_families.",
    )
    scaling_p.add_argument(
        "--subset-seed", type=int, default=0,
        help="Base seed for subset draws; (k, kind) offsets are added.",
    )
    scaling_p.set_defaults(func=cmd_bench_scaling)

    dims_p = bench_sub.add_parser(
        "dims",
        help="Per-dimension covariate decomposition on COSMOS-Web — what does "
             "each Bazaar latent coordinate encode? Emits a long-form "
             "(modality, dim, covariate, r2) parquet.",
    )
    dims_p.add_argument("--D", type=int, default=1024,
                        help="Shared latent dimensionality (default: 1024).")
    dims_p.add_argument("--n-use", type=int, default=45_000,
                        help="Rows to load per modality (default: 45000).")
    dims_p.add_argument("--seed", type=int, default=0,
                        help="Seed for BazaarFit.fit (default: 0).")
    dims_p.add_argument("--out", type=Path, required=True,
                        help="Output parquet path.")
    dims_p.add_argument("--emb-dir", type=Path, default=Path("embeds"),
                        help="Where the per-model .npy embedding cache lives "
                             "(default: ./embeds).")
    dims_p.add_argument("--stream-script", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "scripts" / "stream_embeddings_to_npy.py",
                        help="Path to the embedding downloader script.")
    dims_p.set_defaults(func=cmd_bench_dims)

    probes_p = bench_sub.add_parser(
        "probes",
        help="Per-modality 3×3 probe-direction cosine matrices on COSMOS-Web "
             "(PU Fig 4 replication). Emits (modality, source, prop_i, prop_j, "
             "cos) for Bazaar S, each basket member, and basket-avg.",
    )
    probes_p.add_argument("--D", type=int, default=1024,
                          help="Shared latent dimensionality (default: 1024).")
    probes_p.add_argument("--n-use", type=int, default=45_000,
                          help="Rows to load per modality (default: 45000).")
    probes_p.add_argument("--seed", type=int, default=0,
                          help="Seed for BazaarFit.fit (default: 0).")
    probes_p.add_argument("--out", type=Path, required=True,
                          help="Output parquet path.")
    probes_p.add_argument("--emb-dir", type=Path, default=Path("embeds"),
                          help="Where the per-model .npy embedding cache lives "
                               "(default: ./embeds).")
    probes_p.add_argument("--stream-script", type=Path,
                          default=Path(__file__).resolve().parents[2]
                          / "scripts" / "stream_embeddings_to_npy.py",
                          help="Path to the embedding downloader script.")
    probes_p.set_defaults(func=cmd_bench_probes)

    transfer_p = bench_sub.add_parser(
        "transfer",
        help="Cross-survey generalization (§3.2): fit on source corpus, "
             "probe on target. One parquet per target.",
    )
    transfer_p.add_argument(
        "--target", required=True,
        choices=["cosmos-hsc", "cosmos-jwst", "gz10", "galaxies"],
        help="Target corpus whose labels we probe.",
    )
    transfer_p.add_argument(
        "--sources", type=_parse_sources, default=_parse_sources("all"),
        help="Comma-separated source corpora to fit on (or 'all'). "
             "Each becomes a `fit_source` row in the output. "
             "Include the target itself to get a native baseline at the same "
             "n_fit.",
    )
    transfer_p.add_argument("--n-fit", type=int, default=10_000,
                            help="Rows per source to fit each BazaarFit on. "
                                 "Same for every source so corpus size isn't a confound.")
    transfer_p.add_argument("--D", type=int, default=1024,
                            help="Shared latent dimensionality.")
    transfer_p.add_argument("--n-seeds", type=int, default=5)
    transfer_p.add_argument("--test-size", type=int, default=2_500)
    transfer_p.add_argument("--seed", type=int, default=0,
                            help="Seed threaded into BazaarFit.fit and the concat-PCA fit.")
    transfer_p.add_argument("--emb-dir", type=Path,
                            default=Path("embeds"),
                            help="Per-model .npy cache for cosmos-{hsc,jwst} "
                                 "(default: ./embeds).")
    transfer_p.add_argument("--cache-dir", type=Path, default=None,
                            help="Per-fingerprint embed cache for gz10/galaxies "
                                 "(default: ./embeds).")
    transfer_p.add_argument("--stream-script", type=Path,
                            default=Path(__file__).resolve().parents[2]
                            / "scripts" / "stream_embeddings_to_npy.py",
                            help="Path to the cosmosweb embedding downloader script.")
    transfer_p.add_argument("--target-split", default=None,
                            help="Split for gz10 (default 'train') / galaxies (default 'test'). "
                                 "Ignored for cosmos-*.")
    transfer_p.add_argument("--target-max-samples", type=int, default=None,
                            help="Cap target gz10/galaxies rows (default: full split).")
    transfer_p.add_argument("--target-n-use", type=int, default=None,
                            help="Cap cosmos target rows (default: 45000).")
    transfer_p.add_argument("--source-split", default=None,
                            help="Split for gz10/galaxies sources (defaults match bench commands).")
    transfer_p.add_argument("--source-max-samples", type=int, default=None,
                            help="Cap source gz10/galaxies rows (default: full split, then sliced to n_fit).")
    transfer_p.add_argument("--source-n-use", type=int, default=None,
                            help="Cap cosmos source rows loaded (default: 45000).")
    transfer_p.add_argument("--batch-size", type=int, default=64)
    transfer_p.add_argument("--out", type=Path, required=True,
                            help="Output parquet path (one per target).")
    transfer_p.set_defaults(func=cmd_bench_transfer)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
