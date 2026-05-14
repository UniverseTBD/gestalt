"""Smith42/galaxies ingest (revision v2.0).

`Smith42/galaxies` at `revision="v2.0"` ships DESI Legacy Survey DR8 JPEG
cutouts (`image`, 512×512) alongside the full Walmsley+2023 / LePhare /
MPA-JHU / NSA / OSSY / ALFALFA crossmatched metadata in a single row, so
we read both imagery and the 13 paper-faithful targets from one streaming
pass over one dataset — no second-dataset join required.

We treat the imagery as the `legacysurvey` modality (grz, matching pixel
scale) and pass the PIL RGB through `flux_to_pil`'s pre-rendered branch.

`stream_labels` pulls the 13 regression targets in the same split order;
we sanity-check the first row's `dr8_id` against the imagery stream to
catch any future revision that reorders rows.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterator

import numpy as np
from datasets import load_dataset

from bazaar._ingest.hf_streaming import CatalogSource

GALAXIES_DATASET = "Smith42/galaxies"
GALAXIES_REVISION = "v2.0"
GALAXIES_MODALITY = "legacysurvey"

# Paper-faithful target columns from Sanjaripour+2026 Table 1 (appendix).
# Pairs are treated as differences (e.g. `mag_g_desi - mag_r_desi`). The
# MPA-JHU `total_ssfr_avg` and NSA-derived columns are only populated on
# the cross-matched subset; `run_probe`'s `np.isfinite` mask drops the
# rest before fitting.
GALAXIES_TARGETS: dict[str, tuple[str, ...]] = {
    "mag_abs_g":     ("mag_abs_g_photoz",),
    "mag_abs_z":     ("mag_abs_z_photoz",),
    "g_minus_r":     ("mag_g_desi", "mag_r_desi"),
    "r_minus_z":     ("mag_r_desi", "mag_z_desi"),
    "photo_z":       ("photo_z",),
    "spec_z":        ("spec_z",),
    "mean_ssfr":     ("total_ssfr_avg",),
    "log_mstar":     ("mass_med_photoz",),
    "smooth":        ("smooth-or-featured_smooth_fraction",),
    "disc":          ("smooth-or-featured_featured-or-disk_fraction",),
    "artifact":      ("smooth-or-featured_artifact_fraction",),
    "edge_on":       ("disk-edge-on_yes_fraction",),
    "tight_spiral":  ("spiral-winding_tight_fraction",),
}


def _fingerprint(input: str, revision: str, split: str, max_samples: int | None,
                 modality: str, basket_signature: str) -> str:
    payload = json.dumps(
        ["galaxies", input, revision, split, max_samples, modality, basket_signature],
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


class GalaxiesSource(CatalogSource):
    """CatalogSource that streams JPEG cutouts from Smith42/galaxies v2.0.

    Each row is re-keyed into the shape `embed_basket` expects:
        `{"legacysurvey_image": {"rendered": <H×W×3 uint8 ndarray>}}`
    """

    def rows(self) -> Iterator[dict]:
        ds = load_dataset(
            self.input, revision=GALAXIES_REVISION,
            split=self.split, streaming=True,
        )
        key = f"{self.modality}_image"
        n = 0
        for row in ds:
            pil = row.get("image")
            if pil is None:
                raise KeyError(
                    f"galaxies v2.0 row missing 'image'; "
                    f"got keys {list(row.keys())[:8]!r}..."
                )
            arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
            yield {key: {"rendered": arr}}
            n += 1
            if self.max_samples is not None and n >= self.max_samples:
                return


def galaxies_source(
    *,
    split: str = "test",
    max_samples: int | None = None,
    basket_signature: str = "",
) -> GalaxiesSource:
    """Open Smith42/galaxies (v2.0) as a streamable galaxy catalog."""
    fp = _fingerprint(
        GALAXIES_DATASET, GALAXIES_REVISION, split, max_samples,
        GALAXIES_MODALITY, basket_signature,
    )
    return GalaxiesSource(
        input=GALAXIES_DATASET,
        split=split,
        max_samples=max_samples,
        modality=GALAXIES_MODALITY,
        fingerprint=fp,
    )


def _resolve_target(row: dict, cols: tuple[str, ...]) -> float:
    """Pull `cols` from `row`; subtract pairs (a, b) → a - b; otherwise identity."""
    vals = [row.get(c) for c in cols]
    if any(v is None for v in vals):
        return float("nan")
    if len(cols) == 1:
        v = vals[0]
        return float(v) if v is not None else float("nan")
    if len(cols) == 2:
        a, b = vals
        try:
            return float(a) - float(b)
        except (TypeError, ValueError):
            return float("nan")
    raise ValueError(f"unsupported target column spec: {cols!r}")


def stream_labels(
    split: str = "test",
    max_samples: int | None = None,
    image_dr8_ids: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Stream the 13 paper-faithful targets from Smith42/galaxies v2.0.

    Reads the same dataset/revision as `GalaxiesSource.rows()`, so row
    order is guaranteed by construction. If `image_dr8_ids` is supplied we
    spot-check the first few values anyway — cheap insurance against a
    future revision that quietly reorders rows.
    """
    ds = load_dataset(
        GALAXIES_DATASET, revision=GALAXIES_REVISION,
        split=split, streaming=True,
    )
    buckets: dict[str, list[float]] = {name: [] for name in GALAXIES_TARGETS}
    meta_dr8_ids: list[str] = []
    n = 0
    for row in ds:
        if image_dr8_ids is not None and n < len(image_dr8_ids):
            meta_dr8_ids.append(str(row.get("dr8_id", "")))
        for name, cols in GALAXIES_TARGETS.items():
            buckets[name].append(_resolve_target(row, cols))
        n += 1
        if max_samples is not None and n >= max_samples:
            break

    if image_dr8_ids:
        head = min(len(image_dr8_ids), len(meta_dr8_ids), 5)
        for i in range(head):
            if image_dr8_ids[i] != meta_dr8_ids[i]:
                raise RuntimeError(
                    f"galaxies v2.0 row-order drift detected at index {i}: "
                    f"image dr8_id={image_dr8_ids[i]!r} vs "
                    f"metadata dr8_id={meta_dr8_ids[i]!r}. "
                    f"The dataset's row order has changed; pin a working "
                    f"revision or switch to an explicit dr8_id join."
                )

    return {
        name: np.asarray(buckets[name], dtype=np.float32)
        for name in GALAXIES_TARGETS
    }


def stream_image_dr8_ids(
    split: str = "test",
    max_samples: int | None = None,
) -> list[str]:
    """Return the first-N `dr8_id`s from Smith42/galaxies v2.0 in stream order."""
    ds = load_dataset(
        GALAXIES_DATASET, revision=GALAXIES_REVISION,
        split=split, streaming=True,
    )
    ids: list[str] = []
    for row in ds:
        ids.append(str(row.get("dr8_id", "")))
        if max_samples is not None and len(ids) >= max_samples:
            break
    return ids
