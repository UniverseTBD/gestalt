"""Smoke test for the source-to-target transfer figure."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_transfer_r2_vs_source.py"
_SPEC = importlib.util.spec_from_file_location("plot_transfer_r2_vs_source", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_heatmap_includes_gz10_classification(tmp_path):
    rows = []
    for target in _MODULE.TARGETS:
        kind = "classification" if target == "gz10" else "regression"
        for fit_source in _MODULE.FIT_SOURCES:
            for source in _MODULE.SOURCE_ORDER:
                for seed in range(2):
                    rows.append(
                        {
                            "target": target,
                            "fit_source": fit_source,
                            "source": source,
                            "kind": kind,
                            "seed": seed,
                            "r2": 0.5 + 0.01 * seed if kind == "regression" else float("nan"),
                            "f1": 0.5 + 0.01 * seed if kind == "classification" else float("nan"),
                        }
                    )

    assert len(_MODULE.HEATMAP_ROWS) == 4
    assert all("redshift" not in row[3] for row in _MODULE.HEATMAP_ROWS)
    assert _MODULE.HEATMAP_SOURCES == (
        "basket_mcca_whitened",
        "basket_concat_pca",
    )
    assert _MODULE._lodo_summary_matrix().shape == (4, 2)
    assert _MODULE.COMPARISON_LABELS == (
        "Single-source Gestalt transfer",
        "Cross-survey Gestalt fit",
        "Best-performing model per survey",
    )
    frame = pd.DataFrame(rows)
    assert _MODULE._cross_survey_summary_vector(frame).shape == (4,)
    assert _MODULE._lodo_7k5_summary_vector().shape == (4,)
    assert _MODULE._native_single_summary_vector().shape == (4,)

    setattr(_MODULE, "FIGS", tmp_path)
    _MODULE.plot_mean_grid(frame)
    _MODULE.plot_comparison_strip(frame)

    assert (tmp_path / "transfer_r2_vs_source.pdf").read_bytes()[:4] == b"%PDF"
    assert (tmp_path / "transfer_native_lodo_single_strip.pdf").read_bytes()[:4] == b"%PDF"
