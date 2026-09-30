"""Galaxy Zoo 10 ingest.

`UniverseTBD/mmu_gz10` exposes pre-rendered RGB PNGs (256×256, 0.262"/pix) plus
classification labels and redshifts — *not* the MMU HATS `{flux, band}` shape
the rest of the pipeline expects. We treat the imagery as the `legacysurvey`
modality (closest physical analog: grz, matching pixel scale) and pass the PIL
RGB through `flux_to_pil`'s pre-rendered branch instead of arcsinh-stretching
a flux array.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterator

import numpy as np
from datasets import load_dataset

from gestalt._ingest.hf_streaming import CatalogSource

# TODO(release): unpinned external dataset — pin a commit revision like
# `_ingest/galaxies.py` does.
GZ10_DATASET = "UniverseTBD/mmu_gz10"
GZ10_MODALITY = "legacysurvey"


def _fingerprint(
    input: str, split: str, max_samples: int | None, modality: str, basket_signature: str
) -> str:
    payload = json.dumps(
        ["gz10", input, split, max_samples, modality, basket_signature],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class GZ10Source(CatalogSource):
    """CatalogSource that streams `rgb_image` PNGs from mmu_gz10.

    Each row is re-keyed into the shape `embed_basket` expects:
        `{"legacysurvey_image": {"rendered": <H×W×3 uint8 ndarray>}}`

    The `"rendered"` key is the contract with `flux_to_pil`'s pass-through
    branch; the inner ndarray is plain RGB in standard display order.
    """

    def rows(self) -> Iterator[dict]:
        ds = load_dataset(self.input, split=self.split, streaming=True)
        key = f"{self.modality}_image"
        n = 0
        for row in ds:
            pil = row.get("rgb_image")
            if pil is None:
                raise KeyError(f"gz10 row missing 'rgb_image'; got keys {list(row.keys())!r}")
            arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
            yield {key: {"rendered": arr}}
            n += 1
            if self.max_samples is not None and n >= self.max_samples:
                return


def gz10_source(
    *,
    split: str = "train",
    max_samples: int | None = None,
    basket_signature: str = "",
) -> GZ10Source:
    """Open mmu_gz10 as a streamable galaxy catalog for the basket pipeline."""
    fp = _fingerprint(GZ10_DATASET, split, max_samples, GZ10_MODALITY, basket_signature)
    return GZ10Source(
        input=GZ10_DATASET,
        split=split,
        max_samples=max_samples,
        modality=GZ10_MODALITY,
        fingerprint=fp,
    )


def stream_labels(
    split: str = "train",
    max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Stream gz10_label + redshift columns in the same row order as the embedding pass.

    HF streaming over a fixed dataset+split is deterministic, so the i-th label
    row matches the i-th embedding row produced by `GZ10Source.rows()` for the
    same `max_samples`.
    """
    ds = load_dataset(GZ10_DATASET, split=split, streaming=True)
    labels: list[int] = []
    redshifts: list[float] = []
    for n, row in enumerate(ds):
        labels.append(int(row["gz10_label"]))
        redshifts.append(float(row["redshift"]))
        if max_samples is not None and n + 1 >= max_samples:
            break
    return {
        "gz10_label": np.array(labels, dtype=np.int64),
        "redshift": np.array(redshifts, dtype=np.float32),
    }
