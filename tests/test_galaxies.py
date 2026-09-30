"""Tests for the galaxies bench wiring.

Network-free: we exercise the source-metadata bookkeeping, the
paper-faithful target column spec, and the value-resolution helper that
converts metadata rows into the 13 regression targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from gestalt._ingest.galaxies import (
    GALAXIES_DATASET,
    GALAXIES_MODALITY,
    GALAXIES_REVISION,
    GALAXIES_TARGETS,
    _fingerprint,
    _resolve_target,
    galaxies_source,
)
from gestalt.bench.galaxies import PROPERTIES
from gestalt.bench.linear_probe import run_probe


def test_galaxies_source_metadata():
    src = galaxies_source(split="test", max_samples=100, basket_signature="abc")
    assert src.input == GALAXIES_DATASET
    assert src.split == "test"
    assert src.max_samples == 100
    assert src.modality == GALAXIES_MODALITY
    assert len(src.fingerprint) == 16


def test_galaxies_revision_pinned_to_v2_0():
    # The v2.0 revision is what bakes the metadata columns into the imagery
    # rows. If a future refactor changes this without updating the rest of
    # the module, the bench will start streaming the no-metadata default.
    assert GALAXIES_REVISION == "v2.0"


def test_galaxies_fingerprint_is_deterministic_and_disambiguates():
    a = _fingerprint("ds", "v2.0", "test", 100, "legacysurvey", "sig")
    b = _fingerprint("ds", "v2.0", "test", 100, "legacysurvey", "sig")
    c = _fingerprint("ds", "main", "test", 100, "legacysurvey", "sig")
    d = _fingerprint("ds", "v2.0", "test", 100, "legacysurvey", "other")
    assert a == b
    assert a != c, "revision must contribute to the fingerprint"
    assert a != d, "basket signature must contribute to the fingerprint"
    assert len(a) == 16


def test_paper_targets_match_bench_properties():
    # The bench module's PROPERTIES list should exactly mirror the target
    # spec — drift between the two would mean we silently dropped a target.
    assert set(PROPERTIES) == set(GALAXIES_TARGETS)
    assert len(PROPERTIES) == 13


def test_paper_targets_cover_paper_table():
    # Sanity-check the 13 paper targets are all present by name.
    expected = {
        "mag_abs_g",
        "mag_abs_z",
        "g_minus_r",
        "r_minus_z",
        "photo_z",
        "spec_z",
        "mean_ssfr",
        "log_mstar",
        "smooth",
        "disc",
        "artifact",
        "edge_on",
        "tight_spiral",
    }
    assert set(GALAXIES_TARGETS) == expected


def test_resolve_target_passthrough():
    row = {"photo_z": 0.123, "spec_z": np.float32(0.456)}
    assert _resolve_target(row, ("photo_z",)) == pytest.approx(0.123)
    assert _resolve_target(row, ("spec_z",)) == pytest.approx(0.456)


def test_resolve_target_color_difference():
    row = {"mag_g_desi": 21.0, "mag_r_desi": 20.3, "mag_z_desi": 19.7}
    assert _resolve_target(row, ("mag_g_desi", "mag_r_desi")) == pytest.approx(0.7)
    assert _resolve_target(row, ("mag_r_desi", "mag_z_desi")) == pytest.approx(0.6)


def test_resolve_target_missing_returns_nan():
    row = {"photo_z": None}
    assert np.isnan(_resolve_target(row, ("photo_z",)))
    assert np.isnan(_resolve_target(row, ("mag_g_desi", "mag_r_desi")))  # both absent


def test_run_probe_works_on_galaxies_target_shape():
    # The pipeline's `run_probe` is what the galaxies bench actually calls
    # for every (target, source, seed). Verify it learns a clean signal on
    # the kind of mixed-validity float32 target column we'll produce.
    rng = np.random.default_rng(7)
    N, D = 600, 16
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = (0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.05 * rng.standard_normal(N)).astype(np.float32)
    y[:80] = np.nan  # sparse target — mirrors sSFR/M* coverage
    r2 = run_probe(X, y, test_size=80, random_state=0)
    assert r2 > 0.8
