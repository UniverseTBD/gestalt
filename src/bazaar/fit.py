"""Persistent fit for the basket pipeline.

A `BazaarFit` is everything you need to re-apply the alignment to new data:

- per-model PCA components + means (full-rank per model)
- per-model z-score statistics (post-PCA)
- the MCCA projector V (concatenated whitened features → shared latent)

Fit on a reference corpus once; apply to any new per-model embeddings
forever after via `.transform(...)`.

On-disk layout (under `<fit_dir>/`):

    meta.json                       schema_version, D, seed, basket order
    pca/<family>_<size>.npz         pca_components, pca_mean, zscore_mu, zscore_sd
    mcca.npz                        V (Σ d_m, D)

The per-model PCA runs at native rank (min(d_in, N)) — no per-model dim
reduction. All dim reduction happens at the MCCA SVD, so V's row partitioning
sums the per-model native widths. The JSON pins the basket order so that
partitioning is unambiguous when loading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bazaar.align import mcca_fit, mcca_transform
from bazaar.whiten import pca_zscore_fit, pca_zscore_transform

SCHEMA_VERSION = 4


def _model_key(family: str, size: str) -> str:
    return f"{family}_{size}"


@dataclass
class BazaarFit:
    """Persisted basket fit: per-model full-rank whitener + MCCA projector V."""

    D: int
    seed: int
    basket: list[tuple[str, str]]
    pca: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    mcca_V: np.ndarray | None = None

    @classmethod
    def fit(
        cls,
        embeddings: dict[str, np.ndarray],
        basket: list[tuple[str, str]],
        D: int,
        seed: int = 0,
    ) -> "BazaarFit":
        """Fit on a dict {model_key: (N, d_m) ndarray}.

        `model_key` must be `f"{family}_{size}"` for each (family, size) in
        `basket`. Row order across all entries must match (samples are
        row-aligned).
        """
        keys = [_model_key(f, s) for f, s in basket]
        missing = [k for k in keys if k not in embeddings]
        if missing:
            raise KeyError(f"missing embeddings for {missing}")

        # Per-model PCA goes to native rank min(d_in, N) — full-rank, no
        # per-model dim reduction. MCCA then concatenates the heterogeneous-
        # width views and SVDs the result down to the requested latent dim D.
        pca: dict[str, dict[str, np.ndarray]] = {}
        Zs: list[np.ndarray] = []
        for key in keys:
            E = np.asarray(embeddings[key], dtype=np.float32)
            Z, art = pca_zscore_fit(E, D=E.shape[1], seed=seed)
            pca[key] = art
            Zs.append(Z)

        V, _ = mcca_fit(Zs, D=D, seed=seed)
        return cls(D=D, seed=seed, basket=list(basket), pca=pca, mcca_V=V)

    def transform(self, embeddings: dict[str, np.ndarray]) -> np.ndarray:
        """Project new (or fit) per-model embeddings into the shared latent."""
        if self.mcca_V is None:
            raise RuntimeError("BazaarFit has no MCCA projector — call .fit() first")
        keys = [_model_key(f, s) for f, s in self.basket]
        Zs: list[np.ndarray] = []
        for key in keys:
            if key not in embeddings:
                raise KeyError(f"missing embeddings for {key}")
            E = np.asarray(embeddings[key], dtype=np.float32)
            Zs.append(pca_zscore_transform(E, self.pca[key]))
        return mcca_transform(Zs, self.mcca_V)

    def __call__(self, input, **kwargs) -> np.ndarray:
        """Apply this fit to either pre-computed embeddings or a fresh input.

        - If `input` is a `dict[str, ndarray]` of per-model embeddings, delegate
          to `.transform`.
        - Otherwise treat `input` as a dataset id or path and run the full
          embed + transform pipeline via `bazaar.api.run`.
        """
        if isinstance(input, dict):
            return self.transform(input)
        from bazaar.api import run  # local import: api → fit back-edge
        return run(input, fit=self, **kwargs)

    def save(self, fit_dir: Path | str) -> None:
        fit_dir = Path(fit_dir)
        (fit_dir / "pca").mkdir(parents=True, exist_ok=True)

        meta = {
            "schema_version": SCHEMA_VERSION,
            "D": int(self.D),
            "seed": int(self.seed),
            "basket": [list(t) for t in self.basket],
        }
        (fit_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        for fam, size in self.basket:
            key = _model_key(fam, size)
            art = self.pca[key]
            np.savez(
                fit_dir / "pca" / f"{key}.npz",
                pca_components=art["pca_components"],
                pca_mean=art["pca_mean"],
                zscore_mu=art["zscore_mu"],
                zscore_sd=art["zscore_sd"],
            )

        if self.mcca_V is None:
            raise RuntimeError("nothing to save: BazaarFit.mcca_V is None")
        np.savez(fit_dir / "mcca.npz", V=self.mcca_V)

    @classmethod
    def load(cls, fit_dir: Path | str) -> "BazaarFit":
        fit_dir = Path(fit_dir)
        meta = json.loads((fit_dir / "meta.json").read_text())
        if meta["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {meta['schema_version']} "
                f"(this code only supports {SCHEMA_VERSION}; "
                f"older fits used per-model PCA-to-D, regenerate to load)"
            )
        basket = [tuple(t) for t in meta["basket"]]

        pca: dict[str, dict[str, np.ndarray]] = {}
        for fam, size in basket:
            key = _model_key(fam, size)
            with np.load(fit_dir / "pca" / f"{key}.npz") as f:
                pca[key] = {
                    "pca_components": f["pca_components"],
                    "pca_mean":       f["pca_mean"],
                    "zscore_mu":      f["zscore_mu"],
                    "zscore_sd":      f["zscore_sd"],
                }

        with np.load(fit_dir / "mcca.npz") as f:
            V = f["V"]

        return cls(
            D=int(meta["D"]),
            seed=int(meta["seed"]),
            basket=basket,
            pca=pca,
            mcca_V=V,
        )
