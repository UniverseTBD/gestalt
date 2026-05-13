<p align="center">
  <img src="https://github.com/Smith42/the-bazaar/blob/master/docs/spidey.jpg?raw=true" width="42%">
</p>

# 🛒 The Bazaar 🛒

> *"Given enough eyeballs, all bugs are shallow."* — Linus's Law
>
> *"Given enough aligned foundation models, all astronomy is linear."* — us, probably

A small library + CLI that aligns a heterogeneous basket of frozen
foundation models into one shared embedding via MCCA, ships the fit so you
can apply it to new data with one command, and (separately) tests whether
that basket-mean beats any single model on a downstream linear-probe task.
Currently wired up for COSMOS-Web HSC × JWST imagery and three physical
properties (redshift, log M★, sSFR), but the alignment and probe code is
task-agnostic.

The Bazaar is a stall-by-stall view of representation learning: each
foundation model brings its own goods (its own coordinate system on the
same galaxies), and we ask whether the *aggregate* of the bazaar
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

No upstream-repo dependency — the code here is self-contained (the
embeddings themselves are pulled from
`huggingface.co/datasets/UniverseTBD/pu-embeddings` on first run).

## Use the bazaar on your own data

The headline workflow is one command: a saved MCCA fit projects the
basket-mean shared latent onto whatever you point at.

```bash
# COSMOS-Web embeddings already on HF — apply the shipped fit.
bazaar transform \
  --fit default \
  --emb-dir data/embeddings --modality jwst \
  --out unified.npy
```

Or, if you have raw images that pu's cosmosweb adapter understands and
want the bazaar to embed them for you first:

```bash
bazaar transform \
  --fit default \
  --pu-path /path/to/pu --modality jwst \
  --out unified.npy
```

Roll-your-own fit on a custom corpus:

```bash
bazaar fit \
  --emb-dir data/embeddings --modality jwst --D 256 \
  --out fits/jwst-d256
bazaar transform \
  --fit fits/jwst-d256 \
  --emb-dir data/embeddings --modality jwst \
  --out unified.npy
```

Python API:

```python
from bazaar import BASKET, BazaarFit
fit = BazaarFit.fit(per_model_embeddings, basket=BASKET, D=256)
fit.save("fits/my-fit")
# Later, anywhere:
fit = BazaarFit.load("fits/my-fit")
unified = fit.transform(new_per_model_embeddings)   # (N, D)
```

## Evaluation harness

The benchmark that produced the published results lives behind
`bazaar run`:

```bash
# Embeddings auto-download into data/embeddings/ on first run (~8 GB).
bazaar run --D 256 --out data/results_pca256.parquet
bazaar plot --data data/results_pca256.parquet --suffix _pca256
```

This writes a 1 500-row long-form parquet (modality × property × seed ×
{naive, procrustes, mcca, 22×single}) plus per-modality strip plots, a
2×3 summary grid, and a stats table under `figs/`.

## What this is *not*

- It's **not** an ensemble. The basket members were never trained on the
  downstream task; the combiner sees no labels. See
  [`docs/method.md#vs-ensemble`](docs/method.md#vs-ensemble).
- It's **not** a redshift/mass predictor. It's an evaluation harness
  that measures how much physically-relevant signal lives in the
  *unsupervised shared subspace* across these 22 foundation models.

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE).
