"""Images → per-model embeddings, via a `pu run` subprocess per family.

This is the only place in bazaar that touches parquet. We invoke
`uv run --directory <pu_path> pu run --model <family> --mode <comp_mode>`
once per family, where `comp_mode` is derived from the requested modality
via `COMP_MODE_FOR_MODALITY`. pu writes per-(size) parquets to
`<pu_path>/data/` with columns named `<family>_<size_lstripped>_<modality>`
(per `pu/src/pu/experiments.py:289-300`). We translate those columns into
the existing `.npy` cache layout that `bazaar.basket.emb_npy_path` already
understands, so downstream code reads them with no special-casing.

Supported modalities are pu's image-domain modes that produce a foundation-
model embedding per row: `hsc` and `jwst` (both from pu's cosmosweb jwst
crossmatch) and `legacysurvey` (from `Smith42/legacysurvey_hsc_crossmatched`).
Arbitrary image directories still require a new pu dataset adapter.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

from bazaar.basket import BASKET, emb_npy_path

# Map each user-facing modality to the pu `--mode` value that produces it.
# pu's `--mode <X>` reads `Smith42/<X>_hsc_crossmatched` and emits parquets
# with both `hsc_*` and `<X>_*` columns, so requesting modality=hsc reuses
# the jwst-paired pu run (preserving prior behavior); legacysurvey gets its
# own pu invocation against its own crossmatch.
COMP_MODE_FOR_MODALITY: dict[str, str] = {
    "hsc": "jwst",
    "jwst": "jwst",
    "legacysurvey": "legacysurvey",
}


def _column_for(family: str, size: str, modality: str) -> str:
    """Per `pu/src/pu/experiments.py:292`: <alias>_<size.lstrip('0')>_<mode>."""
    return f"{family}_{size.lstrip('0')}_{modality}".lower()


def _all_cached(basket, modality, cache_dir) -> bool:
    return all(emb_npy_path(cache_dir, modality, f, s).exists() for f, s in basket)


def _family_cached(family, basket, modality, cache_dir) -> bool:
    return all(
        emb_npy_path(cache_dir, modality, f, s).exists()
        for f, s in basket if f == family
    )


def embed_via_pu(
    modality: str,
    pu_path: Path,
    cache_dir: Path,
    basket: list[tuple[str, str]] = BASKET,
    test: bool = False,
) -> dict[str, Path]:
    """Run pu for each family in `basket`, cache .npy per (family, size).

    Returns {f"{family}_{size}": Path(<cached .npy>)} for the requested modality.
    Re-running is idempotent: per-family pu invocations are skipped if all
    sizes for that family are already cached as .npy.
    """
    if modality not in COMP_MODE_FOR_MODALITY:
        raise ValueError(
            f"modality must be one of {sorted(COMP_MODE_FOR_MODALITY)}, got {modality!r}"
        )
    comp_mode = COMP_MODE_FOR_MODALITY[modality]
    cache_dir.mkdir(parents=True, exist_ok=True)
    pu_data_dir = pu_path / "data"
    pu_data_dir.mkdir(parents=True, exist_ok=True)

    families = sorted({f for f, _ in basket})
    for family in families:
        if _family_cached(family, basket, modality, cache_dir):
            print(f"[bazaar.embed] {family}: all sizes cached for {modality}; skipping pu run")
            continue
        cmd = [
            "uv", "run", "--directory", str(pu_path),
            "pu", "run",
            "--model", family,
            "--mode", comp_mode,
        ]
        if test:
            cmd.append("--test")
        print(f"[bazaar.embed] $ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        for fam, size in basket:
            if fam != family:
                continue
            _harvest_parquet(family, size, modality, comp_mode, pu_data_dir, cache_dir)

    return {
        f"{f}_{s}": emb_npy_path(cache_dir, modality, f, s) for f, s in basket
    }


def _harvest_parquet(
    family: str, size: str, modality: str, comp_mode: str,
    pu_data_dir: Path, cache_dir: Path,
) -> None:
    parquet = pu_data_dir / f"{comp_mode}_{family}_{size}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"expected pu output {parquet} after `pu run --model {family} "
            f"--mode {comp_mode}`; got nothing"
        )
    df = pl.read_parquet(parquet)
    col = _column_for(family, size, modality)
    if col not in df.columns:
        raise KeyError(
            f"parquet {parquet} has columns {df.columns}; expected {col!r} "
            f"(check pu naming convention)"
        )
    arr = np.stack([np.asarray(v, dtype=np.float32) for v in df[col].to_list()])
    out = emb_npy_path(cache_dir, modality, family, size)
    np.save(out, arr)
    print(f"[bazaar.embed] {family}_{size} {modality}: harvested {arr.shape} → {out.name}")


def cleanup_pu_data(
    pu_path: Path, modality: str, basket: list[tuple[str, str]] = BASKET,
) -> None:
    """Optionally remove pu's per-run parquet outputs after harvest."""
    comp_mode = COMP_MODE_FOR_MODALITY[modality]
    pu_data_dir = pu_path / "data"
    for fam, size in basket:
        p = pu_data_dir / f"{comp_mode}_{fam}_{size}.parquet"
        if p.exists():
            p.unlink()
    sample_dir = pu_data_dir / "sample_galaxies"
    if sample_dir.exists():
        shutil.rmtree(sample_dir, ignore_errors=True)
