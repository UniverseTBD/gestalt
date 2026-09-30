"""Public Python API for Gestalt.

Three verbs:

- `run(input)` — embed `input` through the 22-model basket and project to a
  unified shared latent via the shipped (or supplied) `GestaltFit`.
- `fit(input)` — embed `input` and fit a fresh `GestaltFit` at the requested D.
- `load(fit_dir_or_repo)` — load a saved `GestaltFit` from a local directory
  or HF repo id (callable: `fit(other_input)`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gestalt._ingest import iter_galaxies
from gestalt.basket import BASKET, basket_signature
from gestalt.embed import embed_basket
from gestalt.fit import GestaltFit

# TODO(release): these resolve `main` at call time — pin to commit revisions
# once the fits are pushed, so a repo update can't silently change results.
DEFAULT_FIT_REPOS: dict[str, str] = {
    "jwst": "UniverseTBD/gestalt-cosmosweb-d256-jwst",
    "hsc": "UniverseTBD/gestalt-cosmosweb-d256-hsc",
}


def _resolve_fit(fit: str | Path | GestaltFit, *, modality: str | None = None) -> GestaltFit:
    if isinstance(fit, GestaltFit):
        return fit
    if str(fit) == "default":
        repo = DEFAULT_FIT_REPOS[modality or "jwst"]
        return GestaltFit.from_pretrained(repo)
    return GestaltFit.from_pretrained(str(fit))


def run(
    input: str | Path,
    *,
    fit: str | Path | GestaltFit = "default",
    split: str = "train",
    max_samples: int | None = None,
    modality: str | None = None,
    cache_dir: Path | None = None,
    batch_size: int = 64,
    out: str | Path | None = None,
) -> np.ndarray:
    """Embed `input` through the basket and apply a `GestaltFit` projection.

    Returns the unified (N, D) shared-latent array. Writes `out` (if given) as
    a `.npy` file alongside.
    """
    sig = basket_signature(BASKET)
    source = iter_galaxies(
        str(input),
        split=split,
        max_samples=max_samples,
        modality=modality,
        basket_signature=sig,
    )
    embeddings = embed_basket(
        source,
        basket=BASKET,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )
    fit_obj = _resolve_fit(fit, modality=source.modality)
    S = fit_obj.transform(embeddings)
    print(f"[gestalt.run] {source.input}  modality={source.modality}  → shared latent {S.shape}")
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, S)
        print(f"[gestalt.run] Wrote {out}")
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
) -> GestaltFit:
    """Embed `input` through the basket and fit a fresh `GestaltFit` at D.

    Returns the fitted object. Saves to `out` (a directory) if given.
    """
    sig = basket_signature(BASKET)
    source = iter_galaxies(
        str(input),
        split=split,
        max_samples=max_samples,
        modality=modality,
        basket_signature=sig,
    )
    embeddings = embed_basket(
        source,
        basket=BASKET,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )
    print(
        f"[gestalt.fit] Fitting GestaltFit(D={D}) on "
        f"{len(BASKET)} models × {next(iter(embeddings.values())).shape[0]} rows "
        f"(modality={source.modality})..."
    )
    fit_obj = GestaltFit.fit(embeddings, basket=BASKET, D=D, seed=seed)
    assert fit_obj.mcca_V is not None
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        fit_obj.save_pretrained(out)
        print(f"[gestalt.fit] Wrote fit → {out}/  (V={fit_obj.mcca_V.shape})")
    return fit_obj


def load(fit_dir_or_repo: str | Path) -> GestaltFit:
    """Load a `GestaltFit` from a local directory or HF repo id.

    The returned object is callable: `fit(input)` is shorthand for
    `gestalt.run(input, fit=fit)`.
    """
    return GestaltFit.from_pretrained(str(fit_dir_or_repo))
