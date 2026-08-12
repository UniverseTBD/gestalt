"""Tests for the COSMOS-Web relation anchor used by Figure 3."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_probe_confusion.py"
_SPEC = importlib.util.spec_from_file_location("plot_probe_confusion", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_catalog_correlations = _MODULE._catalog_correlations


def test_catalog_correlations_match_cosmos_web_labels():
    labels = Path(__file__).parents[1] / "data" / "cosmos_labels.npz"
    correlations = _catalog_correlations(labels)
    np.testing.assert_allclose(
        [correlations[pair] for pair in (
            ("redshift", "mass"),
            ("redshift", "sSFR"),
            ("mass", "sSFR"),
        )],
        [0.2737973, 0.4133363, -0.2450554],
        atol=1e-6,
    )


def test_catalog_correlations_mask_redshift_sentinel_and_clip(tmp_path):
    values = np.arange(10, dtype=np.float64)
    np.savez(
        tmp_path / "labels.npz",
        redshift=np.r_[-99.0, values[1:]],
        mass=2.0 * values + 3.0,
        sSFR=-values,
    )

    correlations = _catalog_correlations(tmp_path / "labels.npz")

    observed = [correlations[pair] for pair in (
        ("redshift", "mass"),
        ("redshift", "sSFR"),
        ("mass", "sSFR"),
    )]
    assert observed[0] > 0.999
    assert observed[1] < -0.999
    np.testing.assert_allclose(observed[2], -1.0, atol=1e-12)
