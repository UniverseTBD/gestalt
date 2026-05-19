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

22 checkpoints across 8 families, all bundled model adapters (no separate
`pu` installation needed):

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

For each foundation model, whiten its frozen embedding — either `zscore`
(per-feature z-score at native width, the default) or `pca_zscore`
(randomised-SVD PCA to D components, then z-score). Stack the M whitened
views into `C = [Ẑ₁ | … | Ẑ_M] ∈ R^{N×MD}` and take its top-D left
singular vectors scaled by their singular values: `S = U Σ`. This is the
**MCCA shared latent** (Carroll/Kettenring MAX-VAR Generalized CCA) and
is the primary output of `bazaar run` / `bazaar fit`. It is entirely
unsupervised — the SVD sees only the per-model features, never the labels.

The fit stores the whitening artifacts and the right-singular-vector matrix
`V = (Vᵀ)ᵀ ∈ R^{MD×D}`, so new data projects as `C_new @ V` without
refitting.

The evaluation benchmark (`bazaar bench cosmos`) additionally compares against
two alignment baselines that require equal-width views (`pca_zscore` mode):

1. **Naive mean** — elementwise mean across the 22 z-scored PCAs (catastrophic
   without alignment — cancels signal across incompatible coordinate systems).
2. **Procrustes / GPA-aligned mean** — iteratively rotate each view onto the
   running consensus mean, then average (`R = U Vᵀ` from `SVD(AᵀB)`).

The headline finding from the COSMOS-Web benchmark: **MCCA wins every cell**.
Procrustes wins every cell except JWST sSFR. Naive mean is worst everywhere.

See [`docs/method.md`](docs/method.md) for the full algorithm details and
[`docs/results.md`](docs/results.md) for the D=128 / D=256 stats.

## Install

```bash
git clone https://github.com/<you>/the-bazaar.git
cd the-bazaar
uv sync          # or: pip install -e .
```

## Use the bazaar on your own data

The top-level API is three verbs. Pass any HF dataset id (or local path)
that `datasets.load_dataset` can open — bazaar streams images directly,
infers the modality from the band list, and handles all embedding and
alignment internally.

### Python

```python
from bazaar import run, fit, load

# Apply the shipped COSMOS-Web fit to any compatible catalog.
embs = run("UniverseTBD/mmu_hsc_pdr3_dud_22.5")          # → (N, 1024) ndarray

# Fit a fresh BazaarFit on your own corpus.
fit_obj = fit("UniverseTBD/mmu_hsc_pdr3_dud_22.5", D=1024, out="fits/mine")

# Reload a saved fit. The fit object is callable — fit_obj(input) is
# shorthand for run(input, fit=fit_obj).
fit_obj = load("fits/mine")
embs = fit_obj("UniverseTBD/some_other_dataset")
```

Lower-level access (if you already have per-model embeddings as ndarrays):

```python
from bazaar import BASKET, BazaarFit

fit_obj = BazaarFit.fit(per_model_embeddings, basket=BASKET, D=256)
fit_obj.save_pretrained("fits/my-fit")

# Later, anywhere — local path, or any HF repo id:
fit_obj = BazaarFit.from_pretrained("fits/my-fit")
fit_obj = BazaarFit.from_pretrained("UniverseTBD/bazaar-cosmosweb-d256-jwst")
unified = fit_obj.transform(new_per_model_embeddings)   # (N, D)

# Publish your own:
fit_obj.push_to_hub("you/your-fit")
```

`BazaarFit` is a `huggingface_hub.ModelHubMixin` — fits are stored as
`config.json` + safetensors files and are Hub-native.

### CLI

```bash
# Embed a catalog and apply the shipped fit (downloads ~8 GB of model weights
# on first run; embeddings are cached under ./embeds).
bazaar run UniverseTBD/mmu_hsc_pdr3_dud_22.5 --out unified.npy

# Fit on your own data.
bazaar fit UniverseTBD/mmu_hsc_pdr3_dud_22.5 --D 256 --out fits/hsc-d256

# Apply a saved fit to new data.
bazaar run UniverseTBD/some_other_dataset --fit fits/hsc-d256 --out new_unified.npy
```

Both `run` and `fit` accept:

| flag | default | description |
|------|---------|-------------|
| `--split` | `train` | dataset split to stream |
| `--max-samples N` | all | cap on galaxies ingested |
| `--modality {hsc,jwst,legacysurvey}` | inferred | override band-set detection |
| `--cache-dir PATH` | `./embeds` | per-model `.npy` cache |
| `--batch-size N` | 64 | inference batch size |

## Evaluation harness

The benchmark that produced the published results lives behind `bazaar bench cosmos`
(kept for reproducibility; requires pre-cached `.npy` embeddings). Sibling
sweeps `bazaar bench gz10` and `bazaar bench galaxies` cover the GZ10
classification + redshift task and the 13 Sanjaripour+2026 regression
targets on Smith42/galaxies respectively:

```bash
bazaar bench cosmos --D 256 --out data/results_pca256.parquet \
                    --emb-dir embeds
bazaar plot         --data data/results_pca256.parquet --suffix _pca256
```

This writes a 1 500-row long-form parquet (modality × property × seed ×
{naive, procrustes, mcca, 22×single}) plus per-modality strip plots, a
2×3 summary grid, and a stats table under `figs/`.

### Cross-survey generalization (`bazaar bench transfer`)

`bench transfer` answers a sharper version of the convergence question:
**is a fit trained on corpus A almost as good on B as B's native fit?** If
yes the recovered subspace is genuinely Platonic; if cross-survey transfer
collapses, the Bazaar claim is *conditional on survey systematics*. For
each target T (`cosmos-hsc`, `cosmos-jwst`, `gz10`, `galaxies`) the sweep
fits a `BazaarFit` (and a `concat→PCA` baseline) on the first `--n-fit`
rows of every source S, transforms T's full embeddings through it, and
runs the same linear probe on T's labels. Source corpora are size-matched
(10 000 rows by default) so "fit corpus size" isn't a confound.

```bash
for t in cosmos-hsc cosmos-jwst gz10 galaxies; do
    bazaar bench transfer --target $t --out data/transfer_$t.parquet
done
```

Each parquet adds a `fit_source` column on top of the standard long-form
schema. The headline cells are the cross-survey ones (`cosmos-hsc ↔
{gz10, galaxies}`); `gz10 ↔ galaxies` is a within-survey control and
`cosmos-hsc ↔ cosmos-jwst` is a cross-modality sanity check.

## What this is *not*

- It's **not** an ensemble. The basket members were never trained on the
  downstream task; the combiner sees no labels. See
  [`docs/method.md#vs-ensemble`](docs/method.md#vs-ensemble).
- It's **not** a redshift/mass predictor. It's an evaluation harness
  that measures how much physically-relevant signal lives in the
  *unsupervised shared subspace* across these 22 foundation models.

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE).
