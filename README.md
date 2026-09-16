<p align="center">
  <img src="https://github.com/Smith42/gestalt/blob/master/docs/spidey.jpg?raw=true" alt="Gestalt" width="42%">
</p>

# 🧩 Gestalt 🧩

Gestalt turns an astronomical image catalog into one joint embedding. It runs
22 frozen foundation models, aligns their feature spaces with MCCA, and returns
a single float32 array of shape `(N, D)`. No labels or task-specific training
are required.

## Install

```bash
git clone https://github.com/Smith42/gestalt.git
cd gestalt
uv sync  # or: pip install -e .
```

## Create a joint embedding

Pass a Hugging Face dataset ID or a local path. Gestalt streams the images,
infers the imaging modality from the bands, embeds every row, and applies the
matching saved alignment fit.

```bash
bazaar run UniverseTBD/mmu_hsc_pdr3_dud_22.5 --out joint.npy
```

The first run downloads the model weights and caches each model's embeddings
under `./embeds`. The output is an `(N, D)` NumPy array.

The same operation is available in Python:

```python
from bazaar import run

joint = run(
    "UniverseTBD/mmu_hsc_pdr3_dud_22.5",
    out="joint.npy",
)  # (N, D) float32 ndarray
```

## Fit an alignment for another corpus

Fit once on a representative catalog, then use that fit to place compatible
new catalogs in the same coordinate system:

```bash
bazaar fit my/catalog --D 1024 --out fits/mine
bazaar run my/new-catalog --fit fits/mine --out joint.npy
```

```python
from bazaar import fit, load

alignment = fit("my/catalog", D=1024, out="fits/mine")
joint = alignment("my/new-catalog")

alignment = load("fits/mine")
joint = alignment("my/new-catalog")
```

Saved fits contain the per-model whitening parameters and MCCA projector. They
can be loaded from a local directory or a Hugging Face Hub repository and can
be published with:

```bash
bazaar push fits/mine you/your-fit
```

## Use existing model embeddings

If the 22 per-model embeddings are already available as NumPy arrays, skip
image inference and work directly with `BazaarFit`:

```python
from bazaar import BASKET, BazaarFit

alignment = BazaarFit.fit(per_model_embeddings, basket=BASKET, D=1024)
alignment.save_pretrained("fits/mine")

alignment = BazaarFit.from_pretrained("fits/mine")
joint = alignment.transform(new_per_model_embeddings)
```

## How the joint embedding is built

Each model's frozen embedding is reduced and standardized independently. The
whitened views are concatenated, and randomized SVD extracts their top `D`
shared directions—the MAX-VAR MCCA joint embedding. A saved fit stores the
right-singular-vector projector, so new rows transform into the same space
without refitting.

See [`docs/method.md`](docs/method.md) for the derivation and fit format.

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE).
