"""Benchmark suite for The Bazaar.

Not part of the install surface: hosts the per-dataset MCCA-vs-single-model
linear-probe sweeps that produced the published results. Use `bazaar bench`
on the CLI, or import the sweeps directly: `bazaar.bench.cosmosweb`,
`bazaar.bench.gz10`, `bazaar.bench.galaxies`, `bazaar.bench.scaling`,
`bazaar.bench.transfer`, `bazaar.bench.dimensions`,
`bazaar.bench.probe_geometry`. The probe evaluators live in
`bazaar.bench.linear_probe`; shared whitening/basket plumbing in
`bazaar.bench._runner`. Plots are rendered by the standalone scripts under
`scripts/plot_*.py`.
"""
