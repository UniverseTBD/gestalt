"""Gestalt — aligned-basket galaxy embeddings.

Public API (three verbs):

    from gestalt import run, fit, load

    # Embed and apply the shipped fit
    embs = run("UniverseTBD/mmu_hsc_pdr3_dud_22.5")

    # Fit a fresh GestaltFit
    fit_obj = fit("UniverseTBD/mmu_hsc_pdr3_dud_22.5", D=1024, out="fits/mine")

    # Reload a saved fit (callable: pass an input to embed + transform)
    fit_obj = load("fits/mine")
    embs = fit_obj("UniverseTBD/some_other_dataset")

Lower-level primitives `mcca_fit`/`mcca_transform` are also exposed for users
who already have per-model embeddings. The benchmark sweep (GPA, naive-mean,
linear probe) lives under `gestalt.bench` and is *not* part of this import
surface; plots are rendered by the standalone `scripts/plot_*.py`.
"""

from gestalt.align import mcca_fit, mcca_transform
from gestalt.api import fit, load, run
from gestalt.basket import BASKET
from gestalt.fit import GestaltFit

__all__ = [
    "BASKET",
    "GestaltFit",
    "fit",
    "load",
    "mcca_fit",
    "mcca_transform",
    "run",
]
