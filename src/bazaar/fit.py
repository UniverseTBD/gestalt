"""Persistent fit for the basket pipeline.

A `BazaarFit` is everything you need to re-apply the alignment to new data:

- per-model PCA components + means (full-rank per model)
- per-model z-score statistics (post-PCA)
- the MCCA projector V (concatenated whitened features → shared latent)

Fit on a reference corpus once; apply to any new per-model embeddings
forever after via `.transform(...)`.

`BazaarFit` inherits `huggingface_hub.ModelHubMixin`, so saved fits behave
like normal HF models: `BazaarFit.from_pretrained("org/repo")` downloads
and loads, `fit.save_pretrained(dir)` writes the standard layout, and
`fit.push_to_hub("org/repo")` publishes.

On-disk layout (under `<fit_dir>/`):

    config.json                       schema_version, D, seed, basket order
    pca/<family>_<size>.safetensors   pca_components, pca_mean, zscore_mu, zscore_sd
    mcca.safetensors                  V (Σ d_m, D)

The per-model PCA runs at native rank (min(d_in, N)) — no per-model dim
reduction. All dim reduction happens at the MCCA SVD, so V's row partitioning
sums the per-model native widths. `config.json` pins the basket order so that
partitioning is unambiguous when loading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from huggingface_hub import ModelHubMixin, snapshot_download
from safetensors.numpy import load_file, save_file

from bazaar.align import mcca_fit, mcca_transform
from bazaar.whiten import pca_zscore_fit, pca_zscore_transform

SCHEMA_VERSION = 5


def _model_key(family: str, size: str) -> str:
    return f"{family}_{size}"


@dataclass
class BazaarFit(ModelHubMixin):
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

    # ----- ModelHubMixin hooks -------------------------------------------------

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory = Path(save_directory)
        (save_directory / "pca").mkdir(parents=True, exist_ok=True)

        config = {
            "schema_version": SCHEMA_VERSION,
            "D": int(self.D),
            "seed": int(self.seed),
            "basket": [list(t) for t in self.basket],
        }
        (save_directory / "config.json").write_text(json.dumps(config, indent=2))

        for fam, size in self.basket:
            key = _model_key(fam, size)
            art = self.pca[key]
            save_file(
                {k: np.ascontiguousarray(v) for k, v in art.items()},
                save_directory / "pca" / f"{key}.safetensors",
            )

        if self.mcca_V is None:
            raise RuntimeError("nothing to save: BazaarFit.mcca_V is None")
        save_file(
            {"V": np.ascontiguousarray(self.mcca_V)},
            save_directory / "mcca.safetensors",
        )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        **model_kwargs,
    ) -> "BazaarFit":
        local = Path(model_id)
        if local.is_dir():
            fit_dir = local
        else:
            fit_dir = Path(snapshot_download(
                repo_id=str(model_id),
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            ))

        config = json.loads((fit_dir / "config.json").read_text())
        if config["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {config['schema_version']} "
                f"(this code only supports {SCHEMA_VERSION}; regenerate the fit)"
            )
        basket = [tuple(t) for t in config["basket"]]

        pca: dict[str, dict[str, np.ndarray]] = {}
        for fam, size in basket:
            key = _model_key(fam, size)
            pca[key] = load_file(fit_dir / "pca" / f"{key}.safetensors")

        V = load_file(fit_dir / "mcca.safetensors")["V"]

        return cls(
            D=int(config["D"]),
            seed=int(config["seed"]),
            basket=basket,
            pca=pca,
            mcca_V=V,
        )
