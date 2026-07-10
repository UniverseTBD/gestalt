# Method

## Pipeline

For each telescope `M ∈ {hsc, jwst}`:

1. **Load** all 22 pre-published `.npy` embeddings `E_m ∈ R^{N × d_m}`
   from `UniverseTBD/pu-embeddings/cosmosweb/`. Native dims `d_m` range
   from 384 (AstroPT-015M) to 5120 (LLaVA-1.5-13B).
2. **PCA + z-score** (`bazaar.whiten.pca_zscore_fit`). Randomised-SVD
   PCA to `min(D, d_m)` components (D = 1024 for the canonical sweep),
   then per-feature z-score on the full N=45 000 rows. Output:
   `Ẑ_m ∈ R^{N × min(D, d_m)}` for each model.
3. **Form two basket sources**:
   - `B_mcca_whitened`: full-rank per-view PCA + z-score, then
     `mcca_fit([Ẑ_1, …, Ẑ_M], D)` returns the projector V and the
     fit-time shared latent S.
   - `B_concat_pca`: PCA-to-D on the raw 22-model horizontal
     concatenation (no per-model whitening) — a baseline that lets the
     SVD pick a global subspace without an alignment step.
4. **Linear probe** (`bazaar.bench.linear_probe.run_probe`) on each basket
   source and each single-model `Ẑ_m`, for `y ∈ {redshift, log M★, sSFR}`
   and 10 random seeds.

Output is long-form parquet: one row per `(modality, property, seed, source)`.

## Alignment primitive

### MCCA (MAX-VAR Generalised CCA)

The Carroll/Kettenring MAX-VAR objective is

```
S* = argmax_{S^T S = I_D}  Σ_m ‖P_m S‖_F²
```

with optimal solution given by the top-D left singular vectors of the
horizontal stack `C = [Ẑ_1 | Ẑ_2 | … | Ẑ_M] ∈ R^{N × MD}`. We use
sklearn's `randomized_svd(C, n_components=D)` and scale the left
singulars by their singular values: `S = U Σ`. This is unsupervised:
it sees only the per-model PCAs, never the labels.

## Persistence and the V matrix

`randomized_svd(C, n_components=D)` returns `(U, σ, Vᵀ)`. The fit-time
shared latent is `S = U Σ`, and equivalently `S = C V` where
`V = (Vᵀ)ᵀ ∈ R^{MD × D}`. So **V alone is enough to project new data
into the same coordinate system**: build `C_new` from new per-model
whitened features and compute `S_new = C_new V`.

`bazaar.fit.BazaarFit` persists everything needed to redo the
preprocessing on new rows:

```
<fit_dir>/
├── config.json                       schema_version, D, seed, basket order
├── pca/<family>_<size>.safetensors   pca_components, pca_mean, zscore_mu, zscore_sd
└── mcca.safetensors                  V (Σ d_m × D)
```

`BazaarFit.transform(new_embeddings)` runs each model's saved PCA + z-score
on the new rows (`(E − μ) @ componentsᵀ`, then `(z − μ_z) / σ_z`),
concatenates the result into `C_new`, and returns `C_new @ V`. On the fit
data itself this differs from `S_fit = U Σ` only by randomized-SVD
reprojection error (~1e-4 absolute on float32). The column space is
preserved exactly.

## <a name="vs-ensemble"></a>How is this different from an ensemble?

Classical ensembles (bagging, stacking, boosting) train base learners
*on the target task* and combine their *predictions*. The Bazaar is
different on four axes:

1. **Combination happens in feature space, not prediction space.** The
   output of the combiner is a new D-dim representation, not a number.
   The same basket serves all properties (redshift, mass, sSFR) without
   re-combining.

2. **The combination is unsupervised.** Procrustes and MCCA see only
   the embeddings — no labels at all. The probe is the only supervised
   step, and it's identical for every property.

3. **The base models aren't redundant — they're heterogeneous.** A
   bagging ensemble averages noisy estimators of the *same* function.
   Our basket members were trained for completely different objectives
   (text-image contrastive, masked pixel reconstruction, video latent
   prediction, JEPA, supervised classification) on completely different
   data. None of them was trained on galaxies. We're not denoising the
   same predictor — we're asking whether fundamentally different
   inductive biases agree on a common subspace.

4. **No training, anywhere except the probe.** PCA + SVD are
   closed-form linear algebra on frozen features. The only learned
   parameters in the pipeline are the linear-probe coefficients.

The closest classical analogue is a **feature-level concatenation
baseline** (concat all 22 embeddings → 8000-d vector → probe). That
would also probably win, but for a different reason: the probe gets to
*supervised*-pick whichever model's coordinates it likes per task. The
Bazaar's MCCA step forces an unsupervised commitment to a shared
subspace *before* the probe sees any labels. That's why a basket win
here is evidence for **representational convergence**; a concat win
would only be evidence that probes are good at feature selection.

## Cross-survey generalization (§3.2)

`bench transfer` operates one rung above the per-corpus sweeps: instead of
training and evaluating a `BazaarFit` on the same corpus, it fits on
corpus A and probes on corpus B's labels. For each (target T, source S),
fit `BazaarFit(D)` on the first `n_fit` rows of S's 22-model embeddings,
project T's full embeddings through it with `BazaarFit.transform`, then
run `run_probe` / `run_classification_probe` on T's labels. A
`concat→PCA-to-D` projector fit on the same S rows runs alongside as a
transferable baseline — without it, a positive transfer result can't be
attributed to MCCA over any unsupervised linear projection. Every source
is capped to the same `n_fit` so corpus size is not a confound across the
matrix (HSC 45k, JWST 45k, GZ10 ~17k, galaxies 86k). Output is one
parquet per target, with the same long-form schema as the per-corpus
sweeps plus a `fit_source` column.

When `S == T` the cell is a *native baseline at the same n_fit* — the
apples-to-apples reference each cross-fit is compared against. The
existing `bench {cosmos,gz10,galaxies}` parquets remain the reference
for "best achievable native R²" at full corpus size.

## Hyperparameters

- `D` (PCA / shared-latent dimensionality): 1024 for all canonical
  results. Per-model PCA truncates to `min(D, d_m)`, so models narrower
  than D (e.g. AstroPT-015M at 384-d) keep their native width. Earlier
  D ∈ {128, 256} sweeps are retained as ablations (`results_pca256.parquet`);
  the D=256 sweep is also where the alignment-mode comparison lives
  (MCCA > Procrustes/GPA ≫ concat→PCA > best single ≫ naive mean).
- `n_seeds`: 10. Each seed picks an independent train/test split via
  `sklearn.model_selection.train_test_split(random_state=seed)`.
- `test_size`: 5 000.
- `n_use`: 45 000 (the full pre-published embedding length).

## Statistical test

Per `(modality, property)`, we compute the paired Wilcoxon signed-rank
test (two-sided, `zero_method="wilcox"`) between each basket source's
10 seed-R²s and the 10 seed-R²s of the best single model (ranked by
mean R²). Reported as `p_white` (`basket_mcca_whitened`) and `p_concat`
(`basket_concat_pca`) in the stats table.
