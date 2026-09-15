"""Tests for the modality registry, dispatch, and rgb passthrough."""

from __future__ import annotations

import numpy as np
import pytest

from bazaar.basket import emb_npy_path
from bazaar.embed.preprocess import flux_to_pil
from bazaar.modalities import (
    HSC,
    JWST,
    LEGACYSURVEY,
    MODALITIES,
    RGB,
    AstroFluxModality,
    infer_from_bands,
)


def test_bundled_registry():
    assert set(MODALITIES) == {"hsc", "jwst", "legacysurvey", "rgb"}
    for name, mod in MODALITIES.items():
        assert mod.name == name
        assert isinstance(mod.ds_tag, str) and mod.ds_tag
    # Singleton instances are the same objects as what the registry returns.
    assert MODALITIES["hsc"] is HSC
    assert MODALITIES["jwst"] is JWST
    assert MODALITIES["legacysurvey"] is LEGACYSURVEY
    assert MODALITIES["rgb"] is RGB


def test_emb_npy_path_uses_modality_ds_tag(tmp_path):
    assert "cosmosweb-hsc-jwst-high-snr-pil2" in emb_npy_path(tmp_path, "hsc", "vit", "base").name
    assert "cosmosweb-hsc-jwst-high-snr-pil2" in emb_npy_path(tmp_path, "jwst", "vit", "base").name
    assert (
        "legacysurvey_hsc_crossmatched"
        in emb_npy_path(tmp_path, "legacysurvey", "vit", "base").name
    )


@pytest.mark.parametrize(
    "bands, expected",
    [
        (["f090w", "f277w", "f444w"], "jwst"),
        (["F090W", "F277W", "F444W"], "jwst"),
        (["g", "r", "i", "z", "y"], "hsc"),
        (["HSC-G", "HSC-R", "HSC-I", "HSC-Z"], "hsc"),
        (["g", "r", "z"], "legacysurvey"),
        (["g", "r", "i", "z", "w1", "w2"], "legacysurvey"),
        (["r", "g", "b"], "rgb"),
        (["R", "G", "B"], "rgb"),
    ],
)
def test_infer_from_bands(bands, expected):
    assert infer_from_bands(bands) == expected


def test_infer_rejects_unknown():
    with pytest.raises(ValueError, match="could not infer modality"):
        infer_from_bands(["foo", "bar"])


def test_infer_rejects_missing():
    with pytest.raises(ValueError, match="missing the 'band'"):
        infer_from_bands(None)


def test_rgb_passthrough_is_lossless_uint8():
    img = np.random.default_rng(0).integers(0, 256, size=(73, 91, 3)).astype(np.uint8)
    out = flux_to_pil({"flux": img}, mode="rgb")
    assert out.dtype == np.uint8
    assert out.shape == (73, 91, 3)
    np.testing.assert_array_equal(out, img)


def test_rgb_passthrough_accepts_chw_and_rendered():
    img = np.zeros((3, 32, 40), dtype=np.uint8)
    img[0] = 10
    img[1] = 20
    img[2] = 30
    out = flux_to_pil({"flux": img}, mode="rgb")
    assert out.shape == (32, 40, 3)
    assert (out[..., 0] == 10).all() and (out[..., 1] == 20).all() and (out[..., 2] == 30).all()

    rendered = np.random.default_rng(1).integers(0, 256, size=(16, 16, 3)).astype(np.uint8)
    np.testing.assert_array_equal(flux_to_pil({"rendered": rendered}, mode="rgb"), rendered)


def test_astro_flux_modality_runs_end_to_end():
    """Smoke test: synthetic flux blobs survive the full astro pipeline."""
    # hsc expects a 4+ band flux cube; pick indices 0,1,3 → grz.
    rng = np.random.default_rng(42)
    hsc_blob = {"flux": rng.uniform(0, 1, size=(5, 160, 160)).astype(np.float32)}
    out = flux_to_pil(hsc_blob, mode="hsc")
    assert out.dtype == np.uint8 and out.shape == (96, 96, 3)

    jwst_blob = {"flux": rng.uniform(0, 1, size=(7, 96, 96)).astype(np.float32)}
    out = flux_to_pil(jwst_blob, mode="jwst")
    assert out.dtype == np.uint8 and out.shape == (96, 96, 3)


def test_unknown_modality_raises():
    with pytest.raises(ValueError, match="unknown modality"):
        flux_to_pil({"flux": np.zeros((3, 10, 10))}, mode="not-a-mode")


def test_astro_flux_modality_is_extensible():
    """Adding a new astronomy survey is a single AstroFluxModality instance."""
    new = AstroFluxModality(
        name="myhsc",
        description="custom",
        bands=("g", "r", "i", "z"),
        ds_tag="custom-tag",
        band_indices=(0, 1, 3),
        norm_bands=("g", "r", "z"),
        resize_extent=(70, 90, 70, 90),
        do_resize=True,
        band_predicate=lambda s: False,
    )
    assert new.ds_tag == "custom-tag"
    assert new.matches_bands({"g", "r", "z"}) is False
