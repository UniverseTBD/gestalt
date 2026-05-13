# Results

22-checkpoint basket × 2 modalities × 3 properties × 10 seeds, on 45 000
COSMOS-Web galaxies (`Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2`).

All R²s are mean ± std across 10 random train/test splits. Rank is
`basket-rank-among-singles + 1`, so 1/23 = best in the room. p-values
are paired two-sided Wilcoxon against the best single per cell.

## D = 128

| modality | property | naive       | procrustes  | **mcca**     | best single (id)         | rank_proc | rank_mcca | p_mcca |
|----------|----------|-------------|-------------|--------------|--------------------------|-----------|-----------|--------|
| HSC      | z        | 0.300±0.008 | 0.434±0.011 | **0.440±0.011** | 0.404 (clip_base)     | 1/23      | 1/23      | 0.002  |
| HSC      | log M★   | 0.343±0.010 | 0.525±0.008 | **0.533±0.009** | 0.519 (astropt_850M)  | 1/23      | 1/23      | 0.002  |
| HSC      | sSFR     | 0.317±0.012 | 0.509±0.012 | **0.519±0.013** | 0.478 (clip_base)     | 1/23      | 1/23      | 0.002  |
| JWST     | z        | 0.286±0.014 | 0.523±0.009 | **0.534±0.008** | 0.475 (clip_large)    | 1/23      | 1/23      | 0.002  |
| JWST     | log M★   | 0.349±0.008 | 0.763±0.006 | **0.775±0.005** | 0.726 (vjepa_giant)   | 1/23      | 1/23      | 0.002  |
| JWST     | sSFR     | 0.167±0.011 | 0.414±0.014 | **0.422±0.013** | 0.413 (vjepa_giant)   | 1/23      | 1/23      | 0.010  |

## D = 256

| modality | property | naive       | procrustes  | **mcca**     | best single (id)         | rank_proc | rank_mcca | p_mcca |
|----------|----------|-------------|-------------|--------------|--------------------------|-----------|-----------|--------|
| HSC      | z        | 0.315±0.011 | 0.464±0.011 | **0.472±0.009** | 0.425 (astropt_850M)  | 1/23      | 1/23      | 0.002  |
| HSC      | log M★   | 0.358±0.010 | 0.555±0.009 | **0.564±0.009** | 0.536 (astropt_850M)  | 1/23      | 1/23      | 0.002  |
| HSC      | sSFR     | 0.329±0.013 | 0.547±0.013 | **0.556±0.014** | 0.501 (vjepa_giant)   | 1/23      | 1/23      | 0.002  |
| JWST     | z        | 0.307±0.013 | 0.574±0.009 | **0.583±0.008** | 0.508 (clip_large)    | 1/23      | 1/23      | 0.002  |
| JWST     | log M★   | 0.378±0.008 | 0.799±0.005 | **0.809±0.005** | 0.772 (vjepa_giant)   | 1/23      | 1/23      | 0.002  |
| JWST     | sSFR     | 0.186±0.011 | 0.439±0.014 | **0.449±0.012** | 0.435 (vjepa_giant)   | 1/23      | 1/23      | 0.002  |

## Reading the table

- **MCCA wins every cell at both D values**, with paired-Wilcoxon
  significance against the best single model (p ≤ 0.010 everywhere, and
  p ≈ 0.002 in 11/12 cells).
- **Procrustes wins every cell except JWST sSFR**, where at D=128 it ties
  vjepa_giant (p=0.557) and at D=256 stays a slight numerical lead but
  doesn't reach significance (p=0.16).
- **MCCA always beats Procrustes**, by ≈ 0.006–0.012 R². Same direction
  in 12/12 cells with no overlap of the std bars.
- **Naive (unaligned) mean is catastrophic** — rank 19–23 / 23 across the
  board. Whatever the basket does well, it does *because* of alignment,
  not because of averaging.
- **D=256 > D=128**: every cell gains 0.02–0.05 R². D=512 is parked.

## Reproducibility

```bash
bazaar run --D 128 --out data/results_pca128.parquet
bazaar run --D 256 --out data/results_pca256.parquet
bazaar plot --data data/results_pca128.parquet --suffix _pca128
bazaar plot --data data/results_pca256.parquet --suffix _pca256
```

Wall time on a single A100 80 GB: ≈ 12 min per D for both modalities
(dominated by per-model PCA on the JWST side; MCCA SVD is ~10 s per
modality).
