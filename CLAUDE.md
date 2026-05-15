# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

The Bazaar is a library + CLI that aligns 22 frozen astronomy foundation-model embeddings (from `UniverseTBD/pu-embeddings`) into one shared latent via MCCA / Procrustes, ships the alignment as a reusable fit, and benchmarks the basket-mean against single-model baselines on a linear probe (redshift, log M★, sSFR for COSMOS-Web HSC×JWST). The alignment and probe code is task-agnostic; the basket and dataset wiring are not.

See `README.md` for the user-facing pitch and `docs/method.md` for the algorithm details (especially the V-matrix derivation that underpins `BazaarFit`).

## Setup, build, test

```bash
uv sync                                 # or: pip install -e .
uv run pytest                           # full test suite
uv run pytest tests/test_fit_transform.py::test_save_load_round_trip   # single test
uv run ruff check src tests             # lint (line-length 100; rules E,F,I)
```

Tests run on small synthetic baskets (no network, no HF download) and are fast.

## CLI surface (`bazaar = bazaar.cli:main`)

- `bazaar run <input> --fit {<dir>|default} --out unified.npy` — embed `<input>` through the 22-model basket and apply a saved fit. `--fit default` downloads the shipped COSMOS-Web D=256 fit from HF.
- `bazaar fit <input> --D 1024 --out fits/<dir>` — embed `<input>` and fit a fresh `BazaarFit` to disk.
- `bazaar bench cosmos --D 256 --out data/results_pca256.parquet --emb-dir data/embeddings` — COSMOS-Web sweep (requires pre-cached `.npy` embeddings; downloads via `scripts/stream_embeddings_to_npy.py` on first run).
- `bazaar bench gz10 --out data/gz10.parquet` — UniverseTBD/mmu_gz10 sweep (classification on `gz10_label` + regression on `redshift`).
- `bazaar bench galaxies --out data/galaxies.parquet` — Smith42/galaxies (v2.0) sweep (13 paper-faithful regression targets).
- `bazaar plot --data <parquet> --suffix _pca256` — strip plots, 2×3 summary grid, stats table into `figs/`.

`run`/`fit` infer modality from the input dataset's band list (override with `--modality`); per-model `.npy` embeddings are cached under `~/.cache/bazaar/embeds`. There is no `bazaar embed` or `bazaar transform` subcommand — the equivalents are `bazaar.embed.embed_basket` and `BazaarFit.transform` in the Python API.

## Architecture

The package is seven top-level modules under `src/bazaar/` plus three sub-packages (`bench/`, `embed/`, `_ingest/`). The data flow is linear:

```
raw images ──(optional: bazaar.embed)──> per-model .npy cache
                                              │
        bazaar.basket.load_embeddings ────────┘
                                              │
                                              ▼
   bazaar.whiten.pca_zscore_fit  (per-model PCA + z-score)
                                              │
                                              ▼
                          bazaar.align.mcca_fit  (shared latent V)
                                              │
                                              ▼
                bazaar.bench.probe.run_probe  (linear probe R²)
```

**`BazaarFit` is the central artifact.** It bundles per-model PCA components + means, per-feature z-score stats, and the MCCA projector `V` of shape `(M·D, D)`. The directory layout is:

```
<fit_dir>/
├── meta.json                # schema_version, D, seed, basket order (pins V's row partitioning)
├── pca/<family>_<size>.npz  # pca_components, pca_mean, zscore_mu, zscore_sd
└── mcca.npz                 # V
```

`transform` projects new data via `C_new @ V`, where `C_new` is the horizontal stack of per-model whitened features. On fit data this differs from the fit-time `S = U Σ` only by randomized-SVD reprojection error (~1e-4 on float32). The column space is preserved exactly.

**Module roles:**

- `basket.py` — the canonical `BASKET` list (22 (family, size) tuples), HF download glue, `load_embeddings`, and `ensure_default_fit_downloaded` (resolves `--fit default`).
- `align.py` — `mcca_fit` (returns `V` + fit-time `S`) and `mcca_transform`. Unsupervised; operates on already-whitened per-model features.
- `whiten.py` — `pca_zscore_fit` / `pca_zscore_transform` (PCA-to-D then per-feature z-score) and `zscore_fit` / `zscore_transform` (no PCA). Tiny and dependency-light so `fit.py` can import it without dragging in the benchmark suite.
- `fit.py` — the persistent `BazaarFit` dataclass. `SCHEMA_VERSION = 4`; bump it if you change the on-disk layout.
- `api.py` — the three Python verbs `run`, `fit`, `load`.
- `cli.py` — argparse subcommands.
- `_ingest/` — HF dataset adapters (`hf_streaming`, `gz10`, `galaxies`) that produce `CatalogSource` rows + label streams.
- `embed/` — per-model embedding pipeline (`embed_basket`), preprocessing, zoom/crop, and model adapters. Vendored from `platonic-universe`.
- `bench/` — `probe.run_probe` / `run_classification_probe` (linear/logistic probes), three dataset sweeps (`cosmosweb.py`, `gz10.py`, `galaxies.py`) on a shared `_runner.py` (whitening + basket-source construction), and `plotting.py` (render-only). All sweeps emit the same long-form schema: `modality, property, kind, source, seed, r2, acc, f1, n_valid` (NaN where the metric is inapplicable to the probe kind).

## Conventions worth knowing

- **Basket order is load-bearing.** `V`'s row partitioning is `[Z_1 | … | Z_M]` in `basket` order; `meta.json` pins this so loaders can't misalign. Always pass the basket explicitly when calling `BazaarFit.fit`/`load_embeddings`.
- **Row alignment across models is assumed, not checked.** `load_embeddings` slices each `.npy` to `n_use` rows and trusts the upstream ordering from `Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2`.
- **Float32 throughout.**
- **Randomized SVD** (sklearn) is used in both PCA and MCCA. Determinism comes from the `seed` arg, not from being exact.
- **Embeddings cache layout:** `<emb_dir>/<telescope>_embeddings_cosmosweb-hsc-jwst-high-snr-pil2_<family>_<size>_45000.npy`. The streaming script downloads parquets and converts in-place to bound peak disk to one embedding at a time.
- **`.gitignore` excludes `data/`, `figs/`, `analysis/`, `embeds/`, `embeds_galaxies/`, and all `*.parquet` / `*.npy`** — those are experiment artefacts and embedding caches, not source.
