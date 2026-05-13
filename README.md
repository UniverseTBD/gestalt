# The Bazaar

> *"Given enough eyeballs, all bugs are shallow."* — Linus's Law
>
> *"Given enough aligned foundation models, all physics is linear."* — us, probably

A small library + CLI that tests whether the **average embedding** from a
heterogeneous basket of frozen foundation models outperforms any single
model on a downstream linear-probe task. Currently configured for
COSMOS-Web HSC × JWST imagery and three physical properties (redshift,
log M★, sSFR), but the alignment and probe code is task-agnostic.

The Bazaar is a stall-by-stall view of representation learning: each
foundation model brings its own goods (its own coordinate system on the
same 45 000 galaxies), and we ask whether the *aggregate* of the bazaar
beats the best single vendor. Spoiler: yes, but only if you align the
stalls first.

## What's in the basket

22 checkpoints across 8 families, all pre-published as per-row embeddings
on `huggingface.co/datasets/UniverseTBD/pu-embeddings/tree/main/cosmosweb`:

| family       | sizes                       | count |
|--------------|-----------------------------|-------|
| AstroPT      | 015M, 095M, 850M            | 3     |
| I-JEPA       | huge, giant                 | 2     |
| V-JEPA       | large, huge, giant          | 3     |
| ViT          | base, large, huge           | 3     |
| ViT-MAE      | base, large, huge           | 3     |
| CLIP         | base, large                 | 2     |
| LLaVA-1.5    | 7b, 13b                     | 2     |
| ConvNeXt-v2  | nano, tiny, base, large     | 4     |

## Method, in one paragraph

For each foundation model, take its frozen 45 000-row embedding, reduce
to D dimensions via randomised-SVD PCA, and z-score each feature. Now
form three candidate "basket" representations:

1. **Naive mean** — straight elementwise mean across the 22 z-scored PCAs.
2. **Procrustes / GPA-aligned mean** — iteratively rotate each model onto
   the running consensus mean, then average. Closed-form per-iteration
   rotation: `R = U V^T` where `U Σ V^T = SVD(A^T B)`.
3. **MCCA shared latent** (Carroll/Kettenring MAX-VAR Generalized CCA) —
   horizontally stack all M PCAs into `C = [Z₁ | … | Z_M] ∈ R^{N×MD}`,
   take the top-D left singular vectors of C scaled by their singular
   values.

Then train a single linear probe (`StandardScaler` + `LinearRegression`)
on each candidate against each physical property, with 1st/99th-percentile
target clipping and a held-out test split of 5 000 galaxies, repeated for
10 random seeds.

The headline finding: **MCCA wins every cell**. Procrustes wins every
cell except JWST sSFR (where it ties the best single). Naive mean is
catastrophic everywhere — averaging across un-aligned per-model PCA
bases cancels signal instead of denoising it.

See [`docs/method.md`](docs/method.md) for the algorithm details and
[`docs/results.md`](docs/results.md) for the full D=128 / D=256 stats.

## Install

```bash
git clone https://github.com/<you>/the-bazaar.git
cd the-bazaar
uv sync          # or: pip install -e .
```

The `pu` (`platonic-universe`) repo is pulled in automatically as a
git-pinned dependency. Only `pu.pu_datasets.cosmosweb.CATALOG_COLUMNS`
is used at runtime, so the import surface is tiny.

## Run

```bash
# Embeddings auto-download into data/embeddings/ on first run (~8 GB).
bazaar run --D 256 --out data/results_pca256.parquet
bazaar plot --data data/results_pca256.parquet --suffix _pca256
```

This will write a 1 500-row long-form parquet (modality × property ×
seed × {naive, procrustes, mcca, 22×single}) plus per-modality strip
plots, a 2×3 summary grid, and a stats table under `figs/`.

## What this is *not*

- It's **not** an ensemble. The basket members were never trained on the
  downstream task; the combiner sees no labels. See
  [`docs/method.md#vs-ensemble`](docs/method.md#vs-ensemble).
- It's **not** a redshift/mass predictor. It's an evaluation harness
  that measures how much physically-relevant signal lives in the
  *unsupervised shared subspace* across these 22 foundation models.

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE).
