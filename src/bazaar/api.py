"""Public Python API for The Bazaar.

Three verbs:

- `run(input)` — embed `input` through the 22-model basket and project to a
  unified shared latent via the shipped (or supplied) `BazaarFit`.
- `fit(input)` — embed `input` and fit a fresh `BazaarFit` at the requested D.
- `load(fit_dir_or_repo)` — load a saved `BazaarFit` from a local directory
  or HF repo id (callable: `fit(other_input)`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bazaar._ingest import iter_galaxies
from bazaar.basket import BASKET, basket_signature
from bazaar.embed import embed_basket
from bazaar.fit import BazaarFit

# TODO(release): these resolve `main` at call time — pin to commit revisions
# once the fits are pushed, so a repo update can't silently change results.
DEFAULT_FIT_REPOS: dict[str, str] = {
    "jwst": "UniverseTBD/bazaar-cosmosweb-d256-jwst",
    "hsc":  "UniverseTBD/bazaar-cosmosweb-d256-hsc",
}


def _resolve_fit(fit: str | Path | BazaarFit, *, modality: str | None = None) -> BazaarFit:
    if isinstance(fit, BazaarFit):
        return fit
    if str(fit) == "default":
        repo = DEFAULT_FIT_REPOS[modality or "jwst"]
        return BazaarFit.from_pretrained(repo)
    return BazaarFit.from_pretrained(str(fit))


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
        fit_obj.save_pretrained(out)
        print(f"[bazaar.fit] Wrote fit → {out}/  (V={fit_obj.mcca_V.shape})")
    return fit_obj


def load(fit_dir_or_repo: str | Path) -> BazaarFit:
    """Load a `BazaarFit` from a local directory or HF repo id.

    The returned object is callable: `fit(input)` is shorthand for
    `bazaar.run(input, fit=fit)`.
    """
    return BazaarFit.from_pretrained(str(fit_dir_or_repo))
