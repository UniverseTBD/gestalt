# Experiment status plan

Scope: every subcommand currently surfaced by `bazaar bench`. Out of scope:
the cross-modal / masked-MCCA ideas in `ideas.md` (those need code that
doesn't exist yet).

## Status table

| Subcommand | Canonical parquet | D | Status | Notes |
|---|---|---|---|---|
| `bench cosmos` | `data/cosmos_1024.parquet` | 1024 | done | D=1024 sweep, `pca_zscore` whiten; basket sources are `basket_mcca_whitened` + `basket_concat_pca` |
| `bench cosmos` (D=256) | `data/results_pca256.parquet` | 256 | done | kept for D comparison |
| `bench cosmos` (z-score ablation) | `data/results_pca1024_zscore_ablation.parquet` | 1024 | done | `pca_zscore` + `zscore`-only whitener side-by-side |
| `bench gz10` | `data/results_pca1024_gz10.parquet` | 1024 | done | classification + redshift regression; scatter in `figs/gz10_r2_vs_model_size.pdf` |
| `bench galaxies` | `data/results_pca1024_galaxies.parquet` | 1024 | done | 13 paper-faithful regression targets; scatter in `figs/galaxies_r2_vs_model_size{,_per_property}.pdf` |
| `bench scaling` | `data/scaling.parquet` | 1024 | done | k ∈ {2, 4, 8, 16, 22}, `basket_mcca` only; curves in `figs/scaling_curves{,_per_property}.pdf` |
| `bench transfer` | `data/transfer_{cosmos-hsc, cosmos-jwst, gz10, galaxies}.parquet` | 1024 | done | 4×4 target × fit_source matrix; n_fit=10000; strip plots in `figs/transfer_{r2,f1}_vs_source*.pdf` |
| `bench dims` | `data/dims.parquet` | 1024 | done | per-dim covariate R²; both modalities |
| `bench probes` | `data/probes_smoke.parquet` | **256** | **todo** | smoke run only — rerun at D=1024; confusion heatmaps in `figs/probes_confusion_smoke.pdf` (will lose `_smoke` suffix after D=1024 rerun) |

## Outstanding work

1. **Probes at D=1024** (the only bench gap).
   ```bash
   UV_CACHE_DIR=/beegfs/general/mjsmith/.uv_cache \
     uv run bazaar bench probes --D 1024 --out data/probes_1024.parquet
   UV_CACHE_DIR=/beegfs/general/mjsmith/.uv_cache \
     uv run scripts/plot_probe_confusion.py
   ```
   Then delete `data/probes_smoke.parquet` + the `_smoke`-suffixed figs in
   `figs/`.

2. **Transfer: add a single-model baseline for the cosmos targets.**
   `transfer_galaxies.parquet` already includes
   `single_astropt_850M_pca_zscore`; the two `transfer_cosmos-*.parquet`
   files only have `basket_mcca_whitened` + `basket_concat_pca`. Adding the
   single-model row would let the cosmos panel mirror the galaxies one.

## Cleanup (delete after you confirm)

These are superseded by entries in the status table above. Suggested
deletions:

- `data/results_pca1024.parquet` (1440 rows) — older partial run of the
  same sweep that `results_pca1024_again.parquet` (1500 rows) replaces.
- `data/cosmos_zscore.parquet`, `data/cosmos_zscore_1024.parquet` — early
  z-score scratch; superseded by `results_pca1024_zscore_ablation.parquet`
  which holds both whitener modes in one file.
- `data/probes_smoke.parquet` — delete *after* the D=1024 rerun lands.
- `figs/probes_{panels,spread,summary,stats}_smoke.{pdf,txt}` — likewise,
  delete after the D=1024 plots replace them.

Total freed: ~30 KB of parquet (negligible on disk; the value is keeping
`data/` unambiguous about which file is "the" cosmos result).

## Rename pass (optional, no compute)

Current `data/` filenames mix conventions (`results_pca1024_*`,
`cosmos_zscore_*`, `transfer_*`). One coherent scheme would be
`<subcommand>_<D>[_<variant>].parquet`:

| from | to |
|---|---|
| `results_pca256.parquet` | `cosmos_256.parquet` |
| `results_pca1024_zscore_ablation.parquet` | `cosmos_1024_zscore_ablation.parquet` |
| `results_pca1024_gz10.parquet` | `gz10_1024.parquet` |
| `results_pca1024_galaxies.parquet` | `galaxies_1024.parquet` |
| `scaling.parquet` | `scaling_1024.parquet` |
| `dims.parquet` | `dims_1024.parquet` |
| `probes_1024.parquet` | (already follows the scheme) |
| `transfer_*.parquet` | (already follows the scheme) |

Worth doing only if you want filenames self-documenting; nothing in the
code reads these paths.
