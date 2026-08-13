"""Tests for the Figure 3 plot."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports]

_SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_probe_confusion.py"
_SPEC = importlib.util.spec_from_file_location("plot_probe_confusion", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_panel_has_one_gestalt_marker_per_pair():
    rows = []
    for pair in _MODULE.PROPERTY_PAIRS:
        for index in range(22):
            rows.append(
                {
                    "modality": "hsc",
                    "source": f"single_model_{index}",
                    "prop_i": pair[0],
                    "prop_j": pair[1],
                    "cos": index / 100,
                }
            )
        rows.append(
            {
                "modality": "hsc",
                "source": "basket_mcca_whitened",
                "prop_i": pair[0],
                "prop_j": pair[1],
                "cos": 0.3,
            }
        )

    fig = plt.figure()
    ax: Any = fig.add_subplot()
    _MODULE._plot_panel(ax, pd.DataFrame(rows), "hsc")
    marker_x = [
        item.get_offsets()[0, 0]
        for item in ax.get_children()
        if hasattr(item, "get_offsets") and item.get_offsets()[0, 0] > 0.5
    ]
    np.testing.assert_allclose(marker_x, [1, 2, 3])
    plt.close(fig)
