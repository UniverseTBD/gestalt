"""Public Python API for The Bazaar.

Three verbs:

- `run(input)` — embed `input` through the 22-model basket and project to a
  unified shared latent via the shipped (or supplied) `BazaarFit`.
- `fit(input)` — embed `input` and fit a fresh `BazaarFit` at the requested D.
- `load(fit_dir)` — load a saved `BazaarFit` (callable: `fit(other_input)`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bazaar._ingest import iter_galaxies
from bazaar.basket import BASKET, basket_signature, ensure_default_fit_downloaded
from bazaar.embed import embed_basket
from bazaar.fit import BazaarFit

DEFAULT_FITS_DIR = Path.home() / ".cache" / "bazaar" / "fits"


def _resolve_fit(fit: str | Path | BazaarFit, *, modality: str | None = None) -> BazaarFit:
    if isinstance(fit, BazaarFit):
        return fit
    if str(fit) == "default":
        DEFAULT_FITS_DIR.mkdir(parents=True, exist_ok=True)
        fit_dir = ensure_default_fit_downloaded(modality or "jwst", DEFAULT_FITS_DIR)
        return BazaarFit.load(fit_dir)
    return BazaarFit.load(Path(fit))


def run(
    input: str | Path,
    *,
    fit: str | Path | BazaarFit = "default",
    split: str = "train",
    max_samples: int | None = None,
    modality: str | None = None,
    cache_dir: Path | None = None,
    batch_size: int = 64,
    out: str | Path | None = None,
) -> np.ndarray:
    """Embed `input` through the basket and apply a `BazaarFit` projection.

    Returns the unified (N, D) shared-latent array. Writes `out` (if given) as
    a `.npy` file alongside.
    """
    sig = basket_signature(BASKET)
    source = iter_galaxies(
        str(input), split=split, max_samples=max_samples,
        modality=modality, basket_signature=sig,
    )
    embeddings = embed_basket(
        source, basket=BASKET, cache_dir=cache_dir, batch_size=batch_size,
    )
    fit_obj = _resolve_fit(fit, modality=source.modality)
    S = fit_obj.transform(embeddings)
    print(f"[bazaar.run] {source.input}  modality={source.modality}  "
          f"→ shared latent {S.shape}")
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, S)
        print(f"[bazaar.run] Wrote {out}")
    return S


def fit(
    input: str | Path,
    *,
    D: int = 1024,
    seed: int = 0,
    split: str = "train",
    max_samples: int | None = None,
    modality: str | None = None,
    cache_dir: Path | None = None,
    batch_size: int = 64,
    out: str | Path | None = None,
) -> BazaarFit:
    """Embed `input` through the basket and fit a fresh `BazaarFit` at D.

    Returns the fitted object. Saves to `out` (a directory) if given.
    """
    sig = basket_signature(BASKET)
    source = iter_galaxies(
        str(input), split=split, max_samples=max_samples,
        modality=modality, basket_signature=sig,
    )
    embeddings = embed_basket(
        source, basket=BASKET, cache_dir=cache_dir, batch_size=batch_size,
    )
    print(f"[bazaar.fit] Fitting BazaarFit(D={D}) on "
          f"{len(BASKET)} models × {next(iter(embeddings.values())).shape[0]} rows "
          f"(modality={source.modality})...")
    fit_obj = BazaarFit.fit(embeddings, basket=BASKET, D=D, seed=seed)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        fit_obj.save(out)
        print(f"[bazaar.fit] Wrote fit → {out}/  (V={fit_obj.mcca_V.shape})")
    return fit_obj


def load(fit_dir: str | Path) -> BazaarFit:
    """Load a `BazaarFit` from disk.

    The returned object is callable: `fit(input)` is shorthand for
    `bazaar.run(input, fit=fit)`.
    """
    return BazaarFit.load(Path(fit_dir))
