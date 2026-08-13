"""Tests for the Figure 1b scaling plot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]

_SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_scaling_curves.py"
_SPEC = importlib.util.spec_from_file_location("plot_scaling_curves", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_one_per_family_curve_stops_at_eight():
    rows = [
        {"modality": "hsc", "property": "redshift", "source": _MODULE.SOURCE,
         "subset_kind": kind, "k": k, "subset_id": 0, "seed": 0, "r2": 0.5}
        for kind, k in [("one_per_family", 8), ("full", 22)]
    ]

    ks, _ = _MODULE._scaling_curve(
        pd.DataFrame(rows), "hsc", ["redshift"], "one_per_family"
    )

    assert ks.tolist() == [8]
