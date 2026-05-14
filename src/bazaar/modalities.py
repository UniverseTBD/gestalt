"""Per-modality image preprocessing pipeline.

A modality bundles everything that differs between input sources:

- ``name`` / ``description`` / ``bands`` — identity and documentation.
- ``ds_tag`` — the dataset tag baked into the ``.npy`` cache filename.
- ``preprocess(blob, ...)`` — flux/PIL blob → ``uint8`` ``(H, W, 3)`` array, ready
  for the model's own autoprocessor.
- ``matches_bands(observed)`` — predicate used to auto-infer the modality from
  a row's ``band`` list when the caller does not pass ``modality=`` explicitly.

The four shipped modalities are ``HSC``, ``JWST``, ``LEGACYSURVEY``, and ``RGB``.
The first three share ``AstroFluxModality`` (band-pick → optional galaxy-fit
resize → per-band arcsinh/linear stretch with global percentiles → BGR-flip →
uint8). ``RGB`` is a pure pass-through for plain 3-band JPG/PNG imagery.

Add your own by constructing a ``Modality`` (typically ``AstroFluxModality``)
and passing it to ``register()``; new astronomy surveys also need an entry in
``embed/data/percentiles.json`` keyed by ``name`` if they use the global
``arcsinh``/``linear`` stretches.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Callable, Protocol

import numpy as np

# `bazaar.embed.zoom` is imported lazily inside `AstroFluxModality.preprocess`
# to avoid an import cycle: `bazaar.embed.__init__` pulls in `bazaar._ingest`,
# which in turn imports this module for `infer_from_bands`.


# ---------------------------------------------------------------------------
# Percentile loader (shared global p1/p99 per (modality, band))
# ---------------------------------------------------------------------------

def _percentiles_path() -> os.PathLike:
    override = os.environ.get("PU_PERCENTILES_PATH") or os.environ.get(
        "BAZAAR_PERCENTILES_PATH"
    )
    if override:
        return override
    return files("bazaar.embed").joinpath("data/percentiles.json")


@lru_cache(maxsize=1)
def _load_percentiles() -> dict:
    with open(_percentiles_path()) as f:
        return json.load(f)


def _norm_consts(mode: str, band_names: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    mode_data = _load_percentiles()[mode]
    return {b: (mode_data[b]["p1"], mode_data[b]["p99"]) for b in band_names}


# ---------------------------------------------------------------------------
# Normalization primitives
# ---------------------------------------------------------------------------

def _arcsinh(chan: np.ndarray, p1: float, p99: float, alpha: float = 20.0) -> np.ndarray:
    t = (chan - p1) / (p99 - p1)
    return (np.arcsinh(alpha * t) / np.arcsinh(alpha)).clip(0, 1)


def _linear(chan: np.ndarray, p1: float, p99: float) -> np.ndarray:
    return ((chan - p1) / (p99 - p1)).clip(0, 1)


def _per_image(chan: np.ndarray, alpha: float = 20.0) -> np.ndarray:
    return _arcsinh(chan, np.percentile(chan, 1), np.percentile(chan, 99), alpha)


# ---------------------------------------------------------------------------
# Modality protocol + concrete implementations
# ---------------------------------------------------------------------------

class Modality(Protocol):
    name: str
    description: str
    bands: tuple[str, ...]
    ds_tag: str

    def preprocess(
        self,
        blob: dict,
        *,
        resize: bool = True,
        resize_mode: str = "match",
        norm_mode: str = "arcsinh",
    ) -> np.ndarray: ...

    def matches_bands(self, observed: set[str]) -> bool: ...


@dataclass(frozen=True)
class AstroFluxModality:
    """Astronomy flux-cube modality: pick 3 bands, resize, stretch, flip."""

    name: str
    description: str
    bands: tuple[str, ...]
    ds_tag: str
    band_indices: tuple[int, int, int]
    norm_bands: tuple[str, str, str]
    resize_extent: tuple[int, int, int, int] | None
    do_resize: bool
    band_predicate: Callable[[set[str]], bool] = field(default=lambda s: False, repr=False)

    def matches_bands(self, observed: set[str]) -> bool:
        return self.band_predicate(observed)

    def preprocess(
        self,
        blob: dict,
        *,
        resize: bool = True,
        resize_mode: str = "match",
        norm_mode: str = "arcsinh",
    ) -> np.ndarray:
        from bazaar.embed.zoom import resize_galaxy_to_fit

        # Pre-rendered RGB (e.g. gz10 PNGs surfaced as legacysurvey): keep as-is,
        # resize to the modality canvas. We do not re-stretch already-display imagery.
        if isinstance(blob, dict) and "rendered" in blob:
            arr = np.asarray(blob["rendered"], dtype=np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(
                    f"'rendered' blob must be (H, W) or (H, W, 3); got shape {arr.shape}"
                )
            if resize and self.do_resize and arr.shape[:2] != (96, 96):
                arr = _resize_to_96(arr)
            return arr

        arr = np.asarray(blob["flux"], np.float32)
        if arr.ndim == 3:
            arr = np.stack([arr[i] for i in self.band_indices], axis=-1)
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            raise ValueError(f"Array shape {arr.shape} for {self.name} not recognised")

        if self.do_resize and resize:
            if resize_mode == "match" and self.resize_extent is not None:
                arr = resize_galaxy_to_fit(arr, force_extent=self.resize_extent, target_size=96)
            else:
                arr = resize_galaxy_to_fit(arr, target_size=96)

        if norm_mode in ("arcsinh", "linear"):
            consts = _norm_consts(self.name, self.norm_bands)
            stretch = _arcsinh if norm_mode == "arcsinh" else _linear
            arr = np.stack(
                [stretch(arr[..., i], *consts[b]) for i, b in enumerate(self.norm_bands)],
                axis=-1,
            )
        elif norm_mode == "per_image":
            arr = np.stack([_per_image(arr[..., i]) for i in range(arr.shape[-1])], axis=-1)

        return (arr[..., ::-1] * 255).astype(np.uint8)


@dataclass(frozen=True)
class RGBModality:
    """Generic 3-band passthrough for JPG/PNG. No band-pick, resize, stretch, or flip."""

    name: str = "rgb"
    description: str = (
        "Generic 3-band passthrough. The model's own autoprocessor handles "
        "final resize and normalization."
    )
    bands: tuple[str, ...] = ("r", "g", "b")
    ds_tag: str = "rgb"

    def matches_bands(self, observed: set[str]) -> bool:
        return observed == {"r", "g", "b"}

    def preprocess(
        self,
        blob: dict,
        *,
        resize: bool = True,
        resize_mode: str = "match",
        norm_mode: str = "arcsinh",
    ) -> np.ndarray:
        src = blob.get("rendered") if isinstance(blob, dict) else None
        if src is None:
            src = blob["flux"]
        arr = np.asarray(src)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"rgb mode expects 3-band imagery; got shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr


def _resize_to_96(arr: np.ndarray) -> np.ndarray:
    """Bilinear zoom to 96x96, preserving channel count and uint8 dtype."""
    from scipy.ndimage import zoom

    zh = 96 / arr.shape[0]
    zw = 96 / arr.shape[1]
    out = zoom(arr.astype(np.float32), (zh, zw, 1), order=1)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HSC = AstroFluxModality(
    name="hsc",
    description="Subaru Hyper Suprime-Cam (ground-based optical, grizy). Shares the cosmosweb crossmatch with JWST.",
    bands=("g", "r", "i", "z", "y"),
    ds_tag="cosmosweb-hsc-jwst-high-snr-pil2",
    band_indices=(0, 1, 3),
    norm_bands=("g", "r", "z"),
    resize_extent=(68, 92, 68, 92),
    do_resize=True,
    band_predicate=lambda s: (
        any("hsc" in b for b in s)
        or s == {"g", "r", "i", "z"}
        or s == {"g", "r", "i", "z", "y"}
    ),
)

JWST = AstroFluxModality(
    name="jwst",
    description="JWST NIRCam (space-based near-IR). Shares the cosmosweb crossmatch with HSC.",
    bands=("f090w", "f277w", "f444w"),
    ds_tag="cosmosweb-hsc-jwst-high-snr-pil2",
    band_indices=(0, 4, 6),
    norm_bands=("f090w", "f277w", "f444w"),
    resize_extent=None,
    do_resize=False,
    band_predicate=lambda s: any(
        b.startswith("f") and b[1:].rstrip("w").isdigit() for b in s
    ),
)

LEGACYSURVEY = AstroFluxModality(
    name="legacysurvey",
    description="DESI Legacy Imaging Surveys (ground-based optical, grz).",
    bands=("g", "r", "z"),
    ds_tag="legacysurvey_hsc_crossmatched",
    band_indices=(0, 1, 3),
    norm_bands=("g", "r", "z"),
    resize_extent=(72, 88, 72, 88),
    do_resize=True,
    band_predicate=lambda s: {"g", "r", "z"} <= s,
)

RGB = RGBModality()

# Order matters for band-set inference: jwst's `f###w` pattern is distinctive,
# hsc with explicit `HSC-` prefix or grizy sets is checked before legacysurvey's
# grz-subset fallback, and rgb is the explicit-only catch-all.
MODALITIES: dict[str, Modality] = {m.name: m for m in (JWST, HSC, LEGACYSURVEY, RGB)}


def get_modality(name: str) -> Modality | None:
    """Return the registered modality, or None if unknown."""
    return MODALITIES.get(name)


def register(modality: Modality) -> None:
    """Add or replace a modality in the global registry."""
    MODALITIES[modality.name] = modality


def infer_from_bands(bands) -> str:
    """Infer a modality name from a list of observed band strings.

    Raises ValueError if no registered modality claims the band set.
    """
    if bands is None:
        raise ValueError("image row is missing the 'band' field — cannot infer modality")
    observed = {str(b).lower() for b in bands}
    for mod in MODALITIES.values():
        if mod.matches_bands(observed):
            return mod.name
    raise ValueError(
        f"could not infer modality from bands {sorted(observed)!r}; "
        f"pass modality= explicitly (one of: {', '.join(MODALITIES)})"
    )
