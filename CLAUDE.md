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

- `bazaar run --D 256 --out data/results.parquet` — full benchmark sweep. Downloads ~8 GB of cached per-model `.npy` embeddings on first run via `scripts/stream_embeddings_to_npy.py`; writes long-form parquet (modality × property × seed × source).
- `bazaar plot --data ... --suffix _pca256` — renders per-modality strip plots, 2×3 summary grid, stats table into `figs/`.
- `bazaar fit --modality {hsc,jwst} --D 256 --out fits/<dir>` — fit a `BazaarFit` and save it.
- `bazaar transform --fit {<dir>|default} --modality ... --out unified.npy` — apply a saved fit. `--fit default` downloads the shipped COSMOS-Web D=256 fit from HF.
- `bazaar embed --modality ... --pu-path /path/to/pu` — pre-warm the per-model embedding cache by shelling out to `pu run`.

All `fit`/`transform`/`embed` commands take either `--emb-dir` (pre-cached `.npy` files) **or** `--pu-path` (will shell out to `pu run --model <family>` once per family and harvest the parquets into the `.npy` cache layout). `PU_PATH` env var is honoured.

## Architecture

The package is intentionally flat — eight modules under `src/bazaar/`. The data flow is linear:

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

- `basket.py` — the canonical `BASKET` list (22 (family, size) tuples), HF download glue, and `load_embeddings`. Also resolves `--fit default` by downloading from `UniverseTBD/pu-embeddings:bazaar-fits/cosmosweb-d256-<modality>/`.
- `align.py` — `mcca_fit` (returns `V` + fit-time `S`) and `mcca_transform`. Unsupervised; operates on already-whitened per-model features.
- `whiten.py` — `pca_zscore_fit` / `pca_zscore_transform` (PCA-to-D then per-feature z-score) and `zscore_fit` / `zscore_transform` (no PCA). Tiny and dependency-light so `fit.py` can import it without dragging in the benchmark suite.
- `bench/cosmosweb.py` — COSMOS-Web benchmark orchestrator (`run_cosmosweb`) over `Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2`. Sibling sweeps live in `bench/gz10.py` (`run_gz10`) and `bench/galaxies.py` (`run_galaxies`).
- `fit.py` — the persistent `BazaarFit` dataclass. `SCHEMA_VERSION = 4`; bump it if you change the on-disk layout.
- `embed.py` — only file that touches parquet; shells out to `uv run --directory <pu_path> pu run --model <family>`, then harvests `<pu_path>/data/<mode>_<family>_<size>.parquet` into the `.npy` cache. Column naming convention is `<family>_<size.lstrip('0')>_<modality>` (pu convention, do not change here).
- `probe.py` — `StandardScaler + LinearRegression`, 1st/99th-percentile clip on targets, `test_size=5000` held out. Numbers must stay directly comparable to upstream `pu`'s regression script.
- `cli.py` — argparse subcommands; default cache dirs are `~/.cache/bazaar/embeds` and `~/.cache/bazaar/fits`.
- `plotting.py` — render-only, no recomputation.

## Conventions worth knowing

- **Basket order is load-bearing.** `V`'s row partitioning is `[Z_1 | … | Z_M]` in `basket` order; `meta.json` pins this so loaders can't misalign. Always pass the basket explicitly when calling `BazaarFit.fit`/`load_embeddings`.
- **Row alignment across models is assumed, not checked.** `load_embeddings` slices each `.npy` to `n_use` rows and trusts the upstream ordering from `Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2`.
- **Float32 throughout.**
- **Randomized SVD** (sklearn) is used in both PCA and MCCA. Determinism comes from the `seed` arg, not from being exact.
- **Embeddings cache layout:** `<emb_dir>/<telescope>_embeddings_cosmosweb-hsc-jwst-high-snr-pil2_<family>_<size>_45000.npy`. The streaming script downloads parquets and converts in-place to bound peak disk to one embedding at a time.
- **`.gitignore` excludes `data/`, `figs/`, `analysis/`, and all `*.parquet` / `*.npy`** — those are experiment artefacts, not source.
