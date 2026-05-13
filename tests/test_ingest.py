"""Tests for `bazaar._ingest.hf_streaming`.

Pure-function checks on modality inference + fingerprint determinism. The
streaming I/O path is exercised by the end-to-end smoke test (which needs
network + GPU and is not run in CI).
"""
from __future__ import annotations

import pytest

from bazaar._ingest.hf_streaming import _fingerprint, _modality_from_bands


@pytest.mark.parametrize("bands, expected", [
    (["f090w", "f277w", "f444w"], "jwst"),
    (["F090W", "F277W", "F444W"], "jwst"),
    (["g", "r", "i", "z", "y"], "hsc"),
    (["HSC-G", "HSC-R", "HSC-I", "HSC-Z"], "hsc"),
    (["g", "r", "z"], "legacysurvey"),
    (["g", "r", "i", "z", "w1", "w2"], "legacysurvey"),
])
def test_modality_inference(bands, expected):
    assert _modality_from_bands(bands) == expected


def test_modality_inference_rejects_unknown():
    with pytest.raises(ValueError, match="could not infer modality"):
        _modality_from_bands(["foo", "bar"])


def test_modality_inference_rejects_missing():
    with pytest.raises(ValueError, match="missing the 'band'"):
        _modality_from_bands(None)


def test_fingerprint_is_deterministic_and_sensitive():
    a = _fingerprint("ds", "train", 256, "hsc", "abc")
    b = _fingerprint("ds", "train", 256, "hsc", "abc")
    c = _fingerprint("ds", "train", 256, "jwst", "abc")
    d = _fingerprint("ds", "train", 257, "hsc", "abc")
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 16
