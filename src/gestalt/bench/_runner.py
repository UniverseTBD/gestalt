"""Shared scaffolding for the per-dataset benchmark sweeps.

The three sweeps (`cosmosweb`, `gz10`, `galaxies`) all do the same thing before
they diverge into dataset-specific probing: per-model whitening, then building
the two basket sources (full-rank whitened MCCA, and concat→PCA-to-D). This
module factors those two steps out so each sweep keeps only its dataset-specific
catalog pass and row-building loop.
"""

from __future__ import annotations

import gc

import numpy as np
from sklearn.decomposition import PCA

from gestalt.align import mcca_fit
from gestalt.whiten import pca_zscore_fit, zscore_fit

WHITEN_MODES: tuple[str, ...] = ("pca_zscore", "zscore")


def whiten_per_model(
    embeddings: dict[str, np.ndarray],
    model_names: list[str],
    D: int,
    whiten_mode: str,
) -> tuple[list[np.ndarray], str]:
    """Build the per-model feature list for single-model probes.

    Returns
    -------
    Zs         : list of (N, d) arrays in `model_names` order.
    single_tag : 'pca{D}' for pca_zscore, 'zscore' for the PCA-ablation mode.
                 Used by callers to label single-model row sources.
    """
    if whiten_mode not in WHITEN_MODES:
        raise ValueError(f"whiten_mode must be one of {WHITEN_MODES}, got {whiten_mode!r}")
    if whiten_mode == "pca_zscore":
        Zs = [pca_zscore_fit(embeddings[name], D=D, seed=0)[0] for name in model_names]
        single_tag = f"pca{D}"
    else:
        Zs = [zscore_fit(embeddings[name])[0] for name in model_names]
        single_tag = "zscore"
    return Zs, single_tag


def build_basket_sources(
    embeddings: dict[str, np.ndarray],
    model_names: list[str],
    D: int,
    *,
    log_prefix: str = "[gestalt]",
) -> list[tuple[str, np.ndarray]]:
    """Build the two canonical basket sources.

    1. `basket_mcca_whitened` — full-rank per-view PCA + z-score, then MCCA→D.
    2. `basket_concat_pca`    — raw concatenation across models, single PCA→D.

    Intermediates (`Zs_white`, `C_raw`) are freed explicitly so peak memory
    stays bounded; the 22-model concatenation can be tens of GB before PCA.
    """
    raw_widths = [embeddings[name].shape[1] for name in model_names]
    print(
        f"{log_prefix} full-rank whitened MCCA on {len(model_names)} models "
        f"(per-model native widths={raw_widths}, sum={sum(raw_widths)}, D={D})..."
    )
    Zs_white = [
        pca_zscore_fit(embeddings[name], D=embeddings[name].shape[1])[0] for name in model_names
    ]
    _, B_mcca_whitened = mcca_fit(Zs_white, D=D, seed=0)
    del Zs_white
    gc.collect()
    print(f"{log_prefix} full-rank whitened MCCA shared latent shape={B_mcca_whitened.shape}")

    print(f"{log_prefix} concat→PCA-to-{D} ablation on {sum(raw_widths)}-d raw concatenation...")
    C_raw = np.concatenate(
        [embeddings[name] for name in model_names],
        axis=1,
    ).astype(np.float32)
    B_concat_pca = (
        PCA(
            n_components=D,
            svd_solver="randomized",
            random_state=0,
        )
        .fit_transform(C_raw)
        .astype(np.float32)
    )
    del C_raw
    gc.collect()
    print(f"{log_prefix} concat→PCA shape={B_concat_pca.shape}")

    return [
        ("basket_mcca_whitened", B_mcca_whitened),
        ("basket_concat_pca", B_concat_pca),
    ]


__all__ = ["WHITEN_MODES", "whiten_per_model", "build_basket_sources"]
