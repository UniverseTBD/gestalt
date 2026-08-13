"""Tests for generated appendix-table structure."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "gen_latex_tables.py"
_SPEC = importlib.util.spec_from_file_location("gen_latex_tables", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_model_order_and_scaling_fit_rows():
    sources = [f"single_{model}_pca1024" for model in reversed(_MODULE.MODEL_LABELS)]
    labels = [_MODULE.clean_source(source) for source in sorted(sources, key=_MODULE.order_key)]

    assert labels == list(_MODULE.MODEL_LABELS.values())
    assert _MODULE.clean_source("basket_mcca_whitened") == r"\textbf{Gestalt}"

    table = _MODULE.render_scaling_fits()
    assert "one/family" not in table
    assert table.count("random &") == 6
