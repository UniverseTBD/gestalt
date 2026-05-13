"""Basket composition and embedding cache.

The default basket is 22 (family, size) checkpoints across 8 foundation-model
families that are already published as per-row .npy embeddings under
`UniverseTBD/pu-embeddings/cosmosweb/`. Each file is keyed by telescope,
family, and size, with 45 000 rows aligned to the COSMOS-Web HSC×JWST
crossmatch (`Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2`).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

BASKET: list[tuple[str, str]] = [
    ("astropt", "015M"), ("astropt", "095M"), ("astropt", "850M"),
    ("ijepa",   "huge"), ("ijepa",   "giant"),
    ("vjepa",   "large"), ("vjepa",  "huge"),  ("vjepa", "giant"),
    ("vit",     "base"), ("vit",     "large"), ("vit",   "huge"),
    ("vit-mae", "base"), ("vit-mae", "large"), ("vit-mae", "huge"),
    ("clip",    "base"), ("clip",    "large"),
    ("llava_15", "7b"),  ("llava_15", "13b"),
    ("convnext", "nano"), ("convnext", "tiny"),
    ("convnext", "base"), ("convnext", "large"),
]

DATASET = "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2"
DS_TAG = DATASET.split("/")[-1]
DOWNLOAD_N_USE = 45_000

# Per-modality dataset tag used in the cache filename. `hsc` and `jwst` share
# the cosmosweb crossmatch (pu produces both views from a single `--mode jwst`
# run); `legacysurvey` lives in pu's `Smith42/legacysurvey_hsc_crossmatched`
# dataset. Unknown modalities fall back to DS_TAG so prior callers still work.
_DS_TAG_BY_MODALITY: dict[str, str] = {
    "hsc": DS_TAG,
    "jwst": DS_TAG,
    "legacysurvey": "legacysurvey_hsc_crossmatched",
}


def emb_npy_path(emb_dir: Path, telescope: str, family: str, size: str) -> Path:
    ds_tag = _DS_TAG_BY_MODALITY.get(telescope, DS_TAG)
    fname = f"{telescope}_embeddings_{ds_tag}_{family}_{size}_{DOWNLOAD_N_USE}.npy"
    return emb_dir / fname


def ensure_default_fit_downloaded(modality: str, cache_dir: Path) -> Path:
    """Download the shipped BazaarFit for `modality` from HF if absent.

    The artifact lives under
    `UniverseTBD/pu-embeddings:bazaar-fits/cosmosweb-d256-<modality>/`
    and contains the `meta.json`, `pca/*.npz`, and `mcca.npz` files written
    by `BazaarFit.save`. Returns the local path to the fit directory.
    """
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    subpath = f"bazaar-fits/cosmosweb-d256-{modality}"
    fit_dir = cache_dir / subpath
    if (fit_dir / "meta.json").exists() and (fit_dir / "mcca.npz").exists():
        return fit_dir

    print(f"[bazaar] Downloading default fit for {modality} to {fit_dir}...")
    snapshot_download(
        repo_id="UniverseTBD/pu-embeddings",
        repo_type="dataset",
        allow_patterns=f"{subpath}/*",
        local_dir=cache_dir,
    )
    if not (fit_dir / "meta.json").exists():
        raise FileNotFoundError(
            f"Download succeeded but {fit_dir/'meta.json'} is missing; "
            f"the default fit may not yet be published."
        )
    return fit_dir


def load_embeddings(
    basket: list[tuple[str, str]],
    telescope: str,
    emb_dir: Path,
    n_use: int = DOWNLOAD_N_USE,
) -> dict[str, np.ndarray]:
    """Load per-model .npy files into a {family_size: (n_use, d_m)} dict."""
    out: dict[str, np.ndarray] = {}
    for fam, size in basket:
        path = emb_npy_path(emb_dir, telescope, fam, size)
        if not path.exists():
            raise FileNotFoundError(f"missing embedding file: {path}")
        E = np.load(path, mmap_mode="r")
        out[f"{fam}_{size}"] = np.asarray(E[:n_use], dtype=np.float32)
        del E
    return out


def ensure_embeddings_downloaded(
    basket: list[tuple[str, str]],
    telescopes: list[str],
    emb_dir: Path,
    stream_script: Path,
) -> None:
    """Download any missing embedding parquets via stream_embeddings_to_npy.py.

    `stream_script` is the path to the standalone downloader script; we keep
    it out of the importable package because it shells out to HF hub and is
    not part of the algorithmic surface.
    """
    missing = [(fam, size, t) for fam, size in basket for t in telescopes
               if not emb_npy_path(emb_dir, t, fam, size).exists()]
    if not missing:
        print(f"[bazaar] All {len(basket) * len(telescopes)} embedding files cached.")
        return

    print(f"[bazaar] {len(missing)} embedding files missing — downloading...")
    subset = ",".join(sorted({f"{fam}:{size}" for fam, size, _ in missing}))
    env = {
        **os.environ,
        "STREAM_REPO":   "UniverseTBD/pu-embeddings",
        "STREAM_FOLDER": "cosmosweb",
        "STREAM_DS_TAG": DS_TAG,
        "STREAM_N_USE":  str(DOWNLOAD_N_USE),
        "STREAM_OUT_DIR": str(emb_dir),
        "STREAM_SUBSET":  subset,
    }
    emb_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(stream_script)]
    subprocess.run(cmd, env=env, check=True)
