"""Tests for the scaling-curve uncertainty."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports]

_SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_scaling_curves.py"
_SPEC = importlib.util.spec_from_file_location("plot_scaling_curves", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_scaling_curve_returns_split_and_draw_standard_deviation():
    rows = [
        {
            "modality": "hsc",
            "property": "redshift",
            "source": _MODULE.SOURCE,
            "subset_kind": "random",
            "subset_id": subset,
            "seed": seed,
            "k": 2,
            "r2": value,
        }
        for subset, seed, value in [(0, 0, 0.2), (0, 1, 0.4), (1, 0, 0.6)]
    ]
    ks, means, stds = _MODULE._scaling_curve(pd.DataFrame(rows), "hsc", ["redshift"], "random")

    np.testing.assert_array_equal(ks, [2])
    np.testing.assert_allclose(means, [0.4])
    np.testing.assert_allclose(stds, [0.2])
