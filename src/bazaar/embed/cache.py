"""Per-(family, size) `.npy` cache for basket embeddings.

The cache directory layout is:

    <cache_dir>/<source_fingerprint>/
      ├── manifest.json
      ├── astropt_015M.npy
      ├── ijepa_huge.npy
      └── …                          # one per (family, size) in BASKET

`source_fingerprint` is a 16-hex SHA-1 derived from
`(input, split, max_samples, modality, basket_signature)` — produced by
`bazaar._ingest.iter_galaxies`. Two runs against the same input + basket +
caps reuse the cache; changing any one invalidates it.

`manifest.json` records what the cache was built from so we can warn on
divergence (different n_rows than expected, mismatched basket order, etc.).
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CACHE_DIR = Path("embeds")


def cache_root(cache_dir: Path | None, fingerprint: str) -> Path:
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return root / fingerprint


def npy_path(root: Path, family: str, size: str) -> Path:
    return root / f"{family}_{size}.npy"


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def write_manifest(root: Path, *, input: str, split: str, modality: str,
                   max_samples: int | None, basket: list[tuple[str, str]],
                   n_rows: int, basket_sig: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": input,
        "split": split,
        "modality": modality,
        "max_samples": max_samples,
        "n_rows": int(n_rows),
        "basket": [list(t) for t in basket],
        "basket_signature": basket_sig,
    }
    manifest_path(root).write_text(json.dumps(payload, indent=2))


def read_manifest(root: Path) -> dict | None:
    p = manifest_path(root)
    if not p.exists():
        return None
    return json.loads(p.read_text())
