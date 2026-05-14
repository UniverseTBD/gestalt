"""Basket composition and embedding cache.

The default basket is 22 (family, size) checkpoints across 8 foundation-model
families. Each entry resolves to a HuggingFace model name (or astropt repo)
via `MODEL_REGISTRY`, which is consumed by `bazaar.embed.embed_basket`.

A legacy `.npy` cache layout (used by `bazaar bench`) is keyed by telescope,
family, size, and a 45 000-row crossmatch — see `emb_npy_path`.
"""
from __future__ import annotations

import hashlib
import json
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

# (family, size) → (adapter_alias, hf_model_name). The adapter alias is what
# `bazaar.embed.models.get_adapter()` keys on; the hf_model_name is what the
# adapter's `load()` passes to `from_pretrained`. Lifted from
# `pu/experiments.py:53-147` for the 8 families in BASKET.
MODEL_REGISTRY: dict[tuple[str, str], tuple[str, str]] = {
    # astropt — all sizes share the same astropt repo; size string selects the checkpoint
    ("astropt", "015M"): ("astropt", "Smith42/astroPT_v2.0"),
    ("astropt", "095M"): ("astropt", "Smith42/astroPT_v2.0"),
    ("astropt", "850M"): ("astropt", "Smith42/astroPT_v2.0"),
    # ijepa
    ("ijepa", "huge"):  ("ijepa", "facebook/ijepa_vith14_22k"),
    ("ijepa", "giant"): ("ijepa", "facebook/ijepa_vitg16_22k"),
    # vjepa
    ("vjepa", "large"): ("vjepa", "facebook/vjepa2-vitl-fpc64-256"),
    ("vjepa", "huge"):  ("vjepa", "facebook/vjepa2-vith-fpc64-256"),
    ("vjepa", "giant"): ("vjepa", "facebook/vjepa2-vitg-fpc64-256"),
    # vit
    ("vit", "base"):  ("vit", "google/vit-base-patch16-224-in21k"),
    ("vit", "large"): ("vit", "google/vit-large-patch16-224-in21k"),
    ("vit", "huge"):  ("vit", "google/vit-huge-patch14-224-in21k"),
    # vit-mae
    ("vit-mae", "base"):  ("vit-mae", "facebook/vit-mae-base"),
    ("vit-mae", "large"): ("vit-mae", "facebook/vit-mae-large"),
    ("vit-mae", "huge"):  ("vit-mae", "facebook/vit-mae-huge"),
    # clip
    ("clip", "base"):  ("clip", "openai/clip-vit-base-patch16"),
    ("clip", "large"): ("clip", "openai/clip-vit-large-patch14"),
    # llava_15
    ("llava_15", "7b"):  ("llava_15", "llava-hf/llava-1.5-7b-hf"),
    ("llava_15", "13b"): ("llava_15", "llava-hf/llava-1.5-13b-hf"),
    # convnext
    ("convnext", "nano"):  ("convnext", "facebook/convnextv2-nano-22k-224"),
    ("convnext", "tiny"):  ("convnext", "facebook/convnextv2-tiny-22k-224"),
    ("convnext", "base"):  ("convnext", "facebook/convnextv2-base-22k-224"),
    ("convnext", "large"): ("convnext", "facebook/convnextv2-large-22k-224"),
}


def basket_signature(basket: list[tuple[str, str]] = BASKET) -> str:
    """Deterministic 16-hex fingerprint of the basket order + composition."""
    return hashlib.sha1(json.dumps(basket).encode()).hexdigest()[:16]

DATASET = "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2"
DS_TAG = DATASET.split("/")[-1]
DOWNLOAD_N_USE = 45_000

def emb_npy_path(emb_dir: Path, telescope: str, family: str, size: str) -> Path:
    """Cache path for one (telescope, family, size) embedding `.npy`.

    The dataset tag comes from the modality registered in `bazaar.modalities`.
    Unknown modalities fall back to `DS_TAG` so ad-hoc callers still work.
    """
    from .modalities import get_modality

    mod = get_modality(telescope)
    ds_tag = mod.ds_tag if mod is not None else DS_TAG
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
