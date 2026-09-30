"""Hugging Face streaming ingest for MultimodalUniverse-style HATS datasets.

The MMU HATS image schema stores per-galaxy multi-band imagery in a single
`image` struct column with `band: list[str]` and `flux: list[list[list[float]]]`
(per-band 2D arrays). This module streams a dataset row-by-row and re-keys
the rows into the `{modality}_image` shape the basket preprocessors expect.

Modality is inferred per-source from the first row's band list (or supplied
explicitly via `modality=...`). The supported modalities map onto the
percentile sets shipped in `gestalt/embed/data/percentiles.json`:

  - "hsc"          — HSC PDR (g, r, i, z[, y]) — picks bands [0,1,3] → grz
  - "jwst"         — JWST NIRCam (f090w … f444w)
  - "legacysurvey" — DECaLS / Legacy Survey (g, r, [i,] z[, w1…w4])
  - "rgb"          — generic 3-band JPG/PNG passthrough (no transforms)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset

from gestalt.modalities import infer_from_bands


@dataclass
class CatalogSource:
    """A streamable catalog of galaxies, ready to feed `embed_basket`.

    Attributes
    ----------
    input : str
        The HF dataset id (or local path) that was opened.
    split : str
    max_samples : int | None
        Cap on rows streamed. `None` means stream everything.
    modality : str
        Inferred or user-supplied band-set modality.
    fingerprint : str
        16-hex deterministic key for the cache directory.
    """

    input: str
    split: str
    max_samples: int | None
    modality: str
    fingerprint: str

    def rows(self) -> Iterator[dict]:
        """Yield `{f"{modality}_image": image_struct}` dicts.

        Streamed lazily; consumers (the embed loop) batch their own DataLoader.
        """
        ds = load_dataset(self.input, split=self.split, streaming=True)
        key = f"{self.modality}_image"
        n = 0
        for row in ds:
            img = row.get("image")
            if img is None:
                raise KeyError(f"row missing 'image' column; got keys {list(row.keys())!r}")
            yield {key: img}
            n += 1
            if self.max_samples is not None and n >= self.max_samples:
                return


def _fingerprint(
    input: str, split: str, max_samples: int | None, modality: str, basket_signature: str
) -> str:
    payload = json.dumps(
        [input, split, max_samples, modality, basket_signature],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def iter_galaxies(
    input: str,
    *,
    split: str = "train",
    max_samples: int | None = None,
    modality: str | None = None,
    basket_signature: str = "",
) -> CatalogSource:
    """Open `input` as a streamable galaxy catalog.

    Peeks the first row to infer modality (unless `modality=` is given), then
    returns a `CatalogSource` that can be iterated repeatedly — each call to
    `.rows()` re-opens the streaming dataset from scratch.
    """
    if modality is None:
        # Peek one row, infer, then re-open in `.rows()`.
        peek = load_dataset(input, split=split, streaming=True)
        first = next(iter(peek))
        img = first.get("image")
        if img is None:
            raise KeyError(
                f"first row of {input!r} missing 'image' column; got keys {list(first.keys())!r}"
            )
        modality = infer_from_bands(img.get("band"))

    fp = _fingerprint(input, split, max_samples, modality, basket_signature)
    return CatalogSource(
        input=input,
        split=split,
        max_samples=max_samples,
        modality=modality,
        fingerprint=fp,
    )
