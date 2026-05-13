"""The Bazaar — aligned-basket evaluation of foundation models.

A small library + CLI for testing whether the *average* embedding from a
heterogeneous basket of frozen foundation models outperforms any single
model on a downstream linear-probe task, after aligning the per-model
representations via Procrustes (GPA) or MCCA.
"""
from bazaar.align import (
    generalized_procrustes,
    mcca_basket,
    mcca_fit,
    mcca_transform,
    orthogonal_procrustes,
)
from bazaar.basket import BASKET
from bazaar.fit import BazaarFit
from bazaar.pipeline import (
    pca_and_zscore,
    pca_zscore_fit,
    pca_zscore_transform,
    run_modality,
)
from bazaar.probe import run_probe

__all__ = [
    "BASKET",
    "BazaarFit",
    "generalized_procrustes",
    "mcca_basket",
    "mcca_fit",
    "mcca_transform",
    "orthogonal_procrustes",
    "pca_and_zscore",
    "pca_zscore_fit",
    "pca_zscore_transform",
    "run_modality",
    "run_probe",
]
