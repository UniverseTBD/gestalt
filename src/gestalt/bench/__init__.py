"""Benchmark suite for Gestalt.

Not part of the install surface: hosts the per-dataset MCCA-vs-single-model
linear-probe sweeps that produced the published results. Use `gestalt bench`
on the CLI, or import the sweeps directly: `gestalt.bench.cosmosweb`,
`gestalt.bench.gz10`, `gestalt.bench.galaxies`, `gestalt.bench.scaling`,
`gestalt.bench.transfer`, `gestalt.bench.dimensions`,
`gestalt.bench.probe_geometry`. The probe evaluators live in
`gestalt.bench.linear_probe`; shared whitening/basket plumbing in
`gestalt.bench._runner`. Plots are rendered by the standalone scripts under
`scripts/plot_*.py`.
"""
