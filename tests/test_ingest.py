"""Tests for `bazaar._ingest.hf_streaming` fingerprint determinism.

Modality inference is now tested in `test_modalities.py` (the inference logic
moved into `bazaar.modalities.infer_from_bands`).
"""
from __future__ import annotations

from bazaar._ingest.hf_streaming import _fingerprint


def test_fingerprint_is_deterministic_and_sensitive():
    a = _fingerprint("ds", "train", 256, "hsc", "abc")
    b = _fingerprint("ds", "train", 256, "hsc", "abc")
    c = _fingerprint("ds", "train", 256, "jwst", "abc")
    d = _fingerprint("ds", "train", 257, "hsc", "abc")
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 16
