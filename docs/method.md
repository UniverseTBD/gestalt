# Method

## Pipeline

For each telescope `M ∈ {hsc, jwst}`:

1. **Load** all 22 pre-published `.npy` embeddings `E_m ∈ R^{N × d_m}`
   from `UniverseTBD/pu-embeddings/cosmosweb/`. Native dims `d_m` range
   from 384 (AstroPT-015M) to 5120 (LLaVA-1.5-13B).
2. **PCA + z-score** (`bazaar.pipeline.pca_and_zscore`). Randomised-SVD
   PCA to `D ∈ {128, 256}` components, then per-feature z-score on the
   full N=45 000 rows. Output: `Ẑ_m ∈ R^{N × D}` for each model.
3. **Form three basket sources**:
   - `B_naive = (1/M) Σ_m Ẑ_m`
   - `B_proc, _, info = generalized_procrustes([Ẑ_1, …, Ẑ_M])`
   - `B_mcca = mcca_basket([Ẑ_1, …, Ẑ_M], D)`
4. **Linear probe** (`bazaar.probe.run_probe`) on each basket source and
   each single-model `Ẑ_m`, for `y ∈ {redshift, log M★, sSFR}` and 10
   random seeds.

Output is long-form parquet: one row per `(modality, property, seed, source)`.

## Alignment primitives

### Orthogonal Procrustes

Given (N, D) matrices `A`, `B` with row-aligned samples, find the
rotation `R ∈ O(D)` minimising `||A R - B||_F`. Closed-form:

```
M = A^T B          # (D, D)
U, Σ, V^T = SVD(M)
R = U V^T
```

Computed in float64 for numerical stability; the result is recast to
the input dtype.

### Generalised Procrustes (GPA)

No privileged reference — iterate:

1. Rotate every matrix onto the running mean.
2. Update the mean to be the average of the rotated matrices.

Stop when the mean stabilises (`tol=1e-6` on the per-iter Frobenius
loss). We do not perform the optional scale or translation steps:
each `Ẑ_m` is already centred (PCA) and on comparable scale (z-score).

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

4. **No training, anywhere except the probe.** PCA + SVD + an
   orthogonal-rotation alignment are all closed-form linear algebra on
   frozen features. The only learned parameters in the pipeline are the
   linear-probe coefficients.

The closest classical analogue is a **feature-level concatenation
baseline** (concat all 22 embeddings → 8000-d vector → probe). That
would also probably win, but for a different reason: the probe gets to
*supervised*-pick whichever model's coordinates it likes per task. The
Bazaar's MCCA/Procrustes step forces an unsupervised commitment to a
shared subspace *before* the probe sees any labels. That's why a
basket win here is evidence for **representational convergence**; a
concat win would only be evidence that probes are good at feature
selection.

## Hyperparameters

- `D` (PCA / shared-latent dimensionality): default 256. We sweep
  {128, 256}; D=512 is parked behind a per-model native-dim floor
  (AstroPT-015M is only 384-d, so D=512 requires either dropping that
  checkpoint or padding).
- `n_seeds`: 10. Each seed picks an independent train/test split via
  `sklearn.model_selection.train_test_split(random_state=seed)`.
- `test_size`: 5 000.
- `n_use`: 45 000 (the full pre-published embedding length).

## Statistical test

Per `(modality, property)`, we compute the paired Wilcoxon signed-rank
test (two-sided, `zero_method="wilcox"`) between each basket source's
10 seed-R²s and the 10 seed-R²s of the best single model (ranked by
mean R²). Reported as `p_naive`, `p_proc`, `p_mcca` in the stats table.
