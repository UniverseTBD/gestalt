"""Tests for the gz10 bench wiring.

Network-free: we exercise the pass-through preprocessor with a synthetic
PIL/numpy image, the classification probe with random embeddings + labels,
and the long-form row shape from a tiny synthetic basket.
"""
from __future__ import annotations

import numpy as np
import pytest

from bazaar._ingest.gz10 import _fingerprint, gz10_source
from bazaar.bench.gz10 import _probe_one
from bazaar.bench.linear_probe import run_classification_probe, run_probe
from bazaar.embed.preprocess import flux_to_pil


def test_flux_to_pil_rendered_short_circuit_resizes():
    arr = (np.linspace(0, 255, 256 * 256 * 3)
           .reshape(256, 256, 3)
           .astype(np.uint8))
    out = flux_to_pil({"rendered": arr}, "legacysurvey", ["legacysurvey"])
    assert out.shape == (96, 96, 3)
    assert out.dtype == np.uint8
    row_means = out.mean(axis=(1, 2))
    assert (np.diff(row_means) >= 0).all(), "pass-through must preserve orientation"


def test_flux_to_pil_rendered_no_resize():
    arr = np.zeros((128, 64, 3), dtype=np.uint8)
    out = flux_to_pil({"rendered": arr}, "legacysurvey", ["legacysurvey"], resize=False)
    assert out.shape == (128, 64, 3)


def test_flux_to_pil_rendered_2d_broadcast():
    arr = np.full((128, 128), 50, dtype=np.uint8)
    out = flux_to_pil({"rendered": arr}, "legacysurvey", ["legacysurvey"])
    assert out.shape == (96, 96, 3)
    assert (out == 50).all()


def test_flux_to_pil_rendered_rejects_bad_shape():
    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        flux_to_pil({"rendered": arr}, "legacysurvey", ["legacysurvey"])


def test_gz10_source_metadata():
    src = gz10_source(split="train", max_samples=100, basket_signature="abc")
    assert src.input == "UniverseTBD/mmu_gz10"
    assert src.split == "train"
    assert src.max_samples == 100
    assert src.modality == "legacysurvey"
    assert len(src.fingerprint) == 16
    other = gz10_source(split="train", max_samples=100, basket_signature="def")
    assert src.fingerprint != other.fingerprint


def test_gz10_fingerprint_is_deterministic():
    a = _fingerprint("ds", "train", 100, "legacysurvey", "sig")
    b = _fingerprint("ds", "train", 100, "legacysurvey", "sig")
    c = _fingerprint("ds", "train", 100, "legacysurvey", "other")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_classification_probe_learns_signal():
    rng = np.random.default_rng(0)
    N, D, K = 2000, 32, 10
    y = rng.integers(0, K, N)
    X = rng.standard_normal((N, D)).astype(np.float32)
    for i, c in enumerate(y):
        X[i, c % D] += 4.0
    acc, f1 = run_classification_probe(X, y, test_size=500, random_state=0)
    assert acc > 0.9
    assert f1 > 0.9


def test_classification_probe_chance_on_random_labels():
    rng = np.random.default_rng(0)
    N, D, K = 2000, 16, 10
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = rng.integers(0, K, N)
    acc, _ = run_classification_probe(X, y, test_size=500, random_state=0)
    assert acc < 0.25


def test_probe_one_dispatches_on_kind():
    rng = np.random.default_rng(1)
    N, D = 500, 8
    X = rng.standard_normal((N, D)).astype(np.float32)
    y_cls = rng.integers(0, 5, N)
    y_reg = X[:, 0].astype(np.float32)
    cls_metrics = _probe_one(X, y_cls, "classification", test_size=100, seed=0)
    reg_metrics = _probe_one(X, y_reg, "regression", test_size=100, seed=0)
    assert set(cls_metrics.keys()) == {"r2", "acc", "f1"}
    assert set(reg_metrics.keys()) == {"r2", "acc", "f1"}
    assert np.isnan(cls_metrics["r2"])
    assert np.isnan(reg_metrics["acc"]) and np.isnan(reg_metrics["f1"])
    assert reg_metrics["r2"] > 0.9


def test_probe_handles_invalid_targets():
    rng = np.random.default_rng(2)
    N, D = 400, 8
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = X[:, 0] + 0.1 * rng.standard_normal(N).astype(np.float32)
    y[:50] = np.nan
    r2 = run_probe(X, y, test_size=80, random_state=0)
    assert r2 > 0.8
