"""The Bazaar — aligned-basket galaxy embeddings.

Public API (three verbs):

    from bazaar import run, fit, load

    # Embed and apply the shipped fit
    embs = run("UniverseTBD/mmu_hsc_pdr3_dud_22.5")

    # Fit a fresh BazaarFit
    fit_obj = fit("UniverseTBD/mmu_hsc_pdr3_dud_22.5", D=1024, out="fits/mine")

    # Reload a saved fit (callable: pass an input to embed + transform)
    fit_obj = load("fits/mine")
    embs = fit_obj("UniverseTBD/some_other_dataset")

Lower-level primitives `mcca_fit`/`mcca_transform` are also exposed for users
who already have per-model embeddings. The benchmark sweep (GPA, naive-mean,
linear probe, plotting) lives under `bazaar.bench` and is *not* part of this
import surface.
"""
from bazaar.align import mcca_fit, mcca_transform
from bazaar.api import fit, load, run
from bazaar.basket import BASKET
from bazaar.fit import BazaarFit

__all__ = [
    "BASKET",
    "BazaarFit",
    "fit",
    "load",
    "mcca_fit",
    "mcca_transform",
    "run",
]
