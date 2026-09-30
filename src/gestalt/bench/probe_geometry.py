"""Per-modality 3×3 probe-direction cosine matrices — `gestalt bench probes`.

Replicates the PU paper's §3.1 ¶3 Fig 4 inside Gestalt S: for each modality,
train linear probes for redshift, log M★, and sSFR; build the 3×3 cosine
matrix between the three probe weight vectors; compare to the same matrix
computed in each single basket member's PCA-whitened embedding space, and
to the basket-elementwise-averaged matrix.

The headline figure is the 4-panel (HSC | JWST × Gestalt | basket-avg) grid.
A clean Gestalt fit should sit *within* the basket spread on each off-diagonal
entry, with a mildly negative sSFR–log M★ correlation (PU §3.1's main-sequence
argument). Wildly outside the spread = MCCA reorganised the §3.1 geometry
rather than preserved it.

Output schema (long-form, one row per (modality, source, prop_i, prop_j)):

    modality, source, prop_i, prop_j, cos, D

`source` is one of:
- `basket_mcca_whitened`  — the Gestalt S latent for this modality.
- `basket_avg`            — elementwise mean of the 22 per-model 3×3s.
- `single_<family>_<size>` — one row block per basket member (22 of these).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gestalt.basket import BASKET, load_embeddings
from gestalt.bench._probe_coeffs import fit_probe_coeffs
from gestalt.bench.cosmosweb import catalog_pass
from gestalt.fit import GestaltFit, _model_key
from gestalt.whiten import pca_zscore_transform

PROPERTIES = ("redshift", "mass", "sSFR")


def _three_probe_cos_matrix(
    Z: np.ndarray,
    labels: dict[str, np.ndarray],
) -> np.ndarray:
    """Train three probes on Z; return their 3×3 cosine matrix.

    Indexing is in the `PROPERTIES` order: row i, col j = cos(w_i, w_j).
    """
    ws = []
    for prop in PROPERTIES:
        ws.append(fit_probe_coeffs(Z, labels[prop])["w"])
    W = np.stack(ws, axis=0)  # (3, D_Z)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    Wn = W / (norms + 1e-12)
    return (Wn @ Wn.T).astype(np.float32)


def _emit_matrix_rows(
    matrix: np.ndarray,
    *,
    modality: str,
    source: str,
    D: int,
) -> list[dict]:
    """Convert a 3×3 cosine matrix to long-form rows."""
    rows: list[dict] = []
    for i, pi in enumerate(PROPERTIES):
        for j, pj in enumerate(PROPERTIES):
            rows.append(
                dict(
                    modality=modality,
                    source=source,
                    prop_i=pi,
                    prop_j=pj,
                    cos=float(matrix[i, j]),
                    D=int(D),
                )
            )
    return rows


def run_probes(
    telescope: str,
    embeddings: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    basket: list[tuple[str, str]],
    *,
    D: int,
    seed: int = 0,
) -> list[dict]:
    """Fit Gestalt; emit 3×3 cosine matrices for Gestalt, per-model, and basket-avg."""
    n_emb = next(iter(embeddings.values())).shape[0]
    n_lab = labels[PROPERTIES[0]].shape[0]
    if n_lab != n_emb:
        raise RuntimeError(
            f"row mismatch: embeddings have {n_emb} rows, labels have "
            f"{n_lab} — streaming order may have drifted."
        )

    print(f"\n[gestalt.probes] === {telescope.upper()} (D={D}) ===")
    print(f"[gestalt.probes] GestaltFit.fit on {n_emb} rows × {len(embeddings)} models")
    fit = GestaltFit.fit(embeddings, basket=basket, D=D, seed=seed)
    S = fit.transform(embeddings).astype(np.float32, copy=False)
    print(f"[gestalt.probes] shared latent S shape={S.shape}")

    rows: list[dict] = []

    # 1. Gestalt S → 3×3 cosine matrix.
    C_gestalt = _three_probe_cos_matrix(S, labels)
    rows.extend(
        _emit_matrix_rows(
            C_gestalt,
            modality=telescope,
            source="basket_mcca_whitened",
            D=D,
        )
    )

    # 2. Per-model: PCA-whiten each basket member to D, train 3 probes,
    #    build its 3×3 matrix. Same evaluation regime as the PU paper
    #    (probes on z-scored per-model PCA features). GestaltFit already
    #    stored full-rank PCA + z-score artifacts during .fit — slicing
    #    the top D PCs reproduces what pca_zscore_fit(E, D=D) would emit
    #    (components are variance-sorted; per-component z-stats are
    #    unaffected by truncation).
    per_model_C: list[np.ndarray] = []
    for fam, size in basket:
        key = _model_key(fam, size)
        art_full = fit.pca[key]
        art_D = {
            "pca_components": art_full["pca_components"][:D],
            "pca_mean": art_full["pca_mean"],
            "zscore_mu": art_full["zscore_mu"][:, :D],
            "zscore_sd": art_full["zscore_sd"][:, :D],
        }
        Z = pca_zscore_transform(embeddings[key], art_D)
        C = _three_probe_cos_matrix(Z, labels)
        per_model_C.append(C)
        rows.extend(
            _emit_matrix_rows(
                C,
                modality=telescope,
                source=f"single_{key}",
                D=D,
            )
        )

    # 3. Basket-averaged 3×3.
    C_basket_avg = np.stack(per_model_C, axis=0).mean(axis=0).astype(np.float32)
    rows.extend(
        _emit_matrix_rows(
            C_basket_avg,
            modality=telescope,
            source="basket_avg",
            D=D,
        )
    )

    _print_summary(C_gestalt, C_basket_avg, per_model_C, telescope)
    return rows


def _print_summary(
    C_gestalt: np.ndarray,
    C_basket_avg: np.ndarray,
    per_model_C: list[np.ndarray],
    telescope: str,
) -> None:
    """Print Gestalt's three off-diagonals alongside the basket spread."""
    pairs = [(0, 1, "z–M★"), (0, 2, "z–sSFR"), (1, 2, "M★–sSFR")]
    print(
        f"[gestalt.probes] {telescope.upper()} 3×3 off-diagonals "
        "(Gestalt vs basket-avg vs basket spread):"
    )
    arr = np.stack(per_model_C, axis=0)
    for i, j, label in pairs:
        baz = float(C_gestalt[i, j])
        avg = float(C_basket_avg[i, j])
        spread = arr[:, i, j]
        lo, hi = float(spread.min()), float(spread.max())
        med = float(np.median(spread))
        within = "in-spread" if lo <= baz <= hi else "OUT-OF-SPREAD"
        print(
            f"  {label:>8s}: gestalt={baz:+.3f}  basket_avg={avg:+.3f}  "
            f"basket_med={med:+.3f}  spread=[{lo:+.3f}, {hi:+.3f}]  ({within})"
        )


def run_probes_cosmos(
    *,
    D: int,
    n_use: int,
    emb_dir: Path,
    telescopes: tuple[str, ...] = ("hsc", "jwst"),
    seed: int = 0,
) -> list[dict]:
    """Top-level entry: catalog pass once, run_probes per telescope."""
    labels = catalog_pass(n_use)
    print(f"[gestalt.probes] labels: {list(labels.keys())} ({n_use} rows each)")

    all_rows: list[dict] = []
    for tele in telescopes:
        embeddings = load_embeddings(BASKET, tele, emb_dir, n_use=n_use)
        all_rows.extend(
            run_probes(
                tele,
                embeddings,
                labels,
                BASKET,
                D=D,
                seed=seed,
            )
        )
        del embeddings
    return all_rows


__all__ = [
    "run_probes",
    "run_probes_cosmos",
    "PROPERTIES",
]
