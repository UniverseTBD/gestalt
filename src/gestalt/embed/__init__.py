"""Image → 22-model basket embeddings.

Vendored from `platonic-universe`: model adapters live in
`gestalt.embed.models`, image preprocessing in `gestalt.embed.preprocess`,
Otsu-based galaxy resize in `gestalt.embed.zoom`, and the on-disk
per-(family, size) `.npy` cache in `gestalt.embed.cache`.

`embed_basket(source, basket, ...)` is the single entrypoint: it walks the
basket in order, loads each foundation model exactly once, and writes one
`.npy` per (family, size) inside `<cache_dir>/<source.fingerprint>/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from datasets import IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from gestalt._ingest import CatalogSource
from gestalt.basket import BASKET, MODEL_REGISTRY, basket_signature
from gestalt.embed.cache import (
    DEFAULT_CACHE_DIR,
    cache_root,
    npy_path,
    write_manifest,
)
from gestalt.embed.models import ModelAdapter, get_adapter

__all__ = [
    "DEFAULT_CACHE_DIR",
    "ModelAdapter",
    "embed_basket",
    "get_adapter",
]


def _missing_cache_entries(root: Path, basket) -> list[tuple[str, str]]:
    return [(f, s) for f, s in basket if not npy_path(root, f, s).exists()]


def _materialise(source: CatalogSource, mode_key: str) -> list[dict]:
    """Pull rows into memory once; we reuse them across 22 models.

    The basket has 22 models; streaming the source 22× would re-download
    the same parquets each time. Held in memory we trade ~MB×N_rows of RAM
    for one-shot ingest.
    """
    return list(source.rows())


def _to_iterable_dataset(rows: list[dict]) -> IterableDataset:
    """Wrap an in-memory row list as a streaming HF dataset that `.map` can hit."""

    def gen():
        yield from rows

    return IterableDataset.from_generator(gen)


def embed_basket(
    source: CatalogSource,
    *,
    basket: list[tuple[str, str]] = BASKET,
    cache_dir: Path | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    enable_amp: bool = True,
    device: str = "cuda",
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Run `source` through each basket model, caching per-model `.npy`.

    Returns `{f"{family}_{size}": (N, d_m) ndarray}` in basket order. Models
    already cached are loaded directly from disk; missing models are
    embedded and saved.
    """
    if device != "cuda":
        raise NotImplementedError("Only CUDA inference is supported today (vendored from pu).")

    root = cache_root(cache_dir, source.fingerprint)
    root.mkdir(parents=True, exist_ok=True)
    sig = basket_signature(basket)

    missing = _missing_cache_entries(root, basket)
    if not missing and not force:
        print(f"[gestalt.embed] All {len(basket)} models cached at {root}")
    else:
        if force:
            missing = list(basket)
        rows = _materialise(source, f"{source.modality}_image")
        n_rows = len(rows)
        print(
            f"[gestalt.embed] {n_rows} rows from {source.input!r} "
            f"(modality={source.modality}); {len(missing)}/{len(basket)} "
            f"models to embed → {root}"
        )

        # Per-family work-share: pu's adapters lazy-load weights inside `load()`,
        # so we instantiate one adapter at a time and run all rows through it.
        for family, size in missing:
            alias, hf_name = MODEL_REGISTRY[(family, size)]
            adapter_cls = get_adapter(alias)
            adapter: ModelAdapter = adapter_cls(hf_name, size, alias=alias)
            adapter.load()
            if enable_amp:
                adapter.enable_amp(True)
            processor = adapter.get_preprocessor(
                [source.modality],
                resize=True,
                resize_mode="match",
            )

            ds = (
                _to_iterable_dataset(rows)
                .map(processor)
                .remove_columns([f"{source.modality}_image"])
            )
            if hasattr(ds, "with_format"):
                ds = ds.with_format("torch")
            dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)  # pyright: ignore[reportArgumentType]  # datasets.IterableDataset is accepted at runtime

            n_batches = (n_rows + batch_size - 1) // batch_size
            zs: list[torch.Tensor] = []
            with torch.no_grad():
                for batch in tqdm(
                    dl,
                    total=n_batches,
                    desc=f"embed {family}_{size}",
                    unit="batch",
                ):
                    zs.append(adapter.embed_for_mode(batch, source.modality).cpu())
            Z = torch.cat(zs).numpy().astype(np.float32)
            out_path = npy_path(root, family, size)
            np.save(out_path, Z)
            print(f"[gestalt.embed]   {family}_{size:8s}  → {Z.shape}  {out_path.name}")

            # Free GPU memory before the next family
            del adapter, ds, dl, zs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_manifest(
            root,
            input=source.input,
            split=source.split,
            modality=source.modality,
            max_samples=source.max_samples,
            basket=basket,
            n_rows=n_rows,
            basket_sig=sig,
        )

    out: dict[str, np.ndarray] = {}
    for f, s in basket:
        out[f"{f}_{s}"] = np.load(npy_path(root, f, s))
    return out
