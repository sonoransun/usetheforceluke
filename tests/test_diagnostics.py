"""Smoke tests for the technical-diagnostic figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from usetheforce.viz.diagnostics import (
    plot_conservation_drift,
    plot_field_heatmap_grid,
    plot_g_effective_crossover,
    plot_yukawa_screening_sweep,
)


def test_g_effective_crossover_returns_figure_with_curve() -> None:
    fig = plot_g_effective_crossover(T_min_mev=10, T_max_mev=500, n=50)
    ax = fig.axes[0]
    # 1 main curve + 2 hlines + 1 vline = 4 line2D objects expected.
    assert len(ax.lines) >= 4
    assert ax.get_xscale() == "log"


def test_yukawa_screening_sweep_one_line_per_lambda_plus_reference() -> None:
    lambdas = [1.0, 10.0, 100.0]
    fig = plot_yukawa_screening_sweep(lambdas_m=lambdas, r_min_m=0.5, r_max_m=1e3, n=80)
    ax = fig.axes[0]
    # one line per λ + one dashed reference line
    assert len(ax.lines) == len(lambdas) + 1
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_conservation_drift_plots_at_least_one_model() -> None:
    fig = plot_conservation_drift(t_periods=0.25, n_eval=80)
    ax = fig.axes[0]
    # 3 model lines + 1 horizontal reference floor expected.
    assert len(ax.lines) >= 2
    assert ax.get_yscale() == "log"


def test_field_heatmap_grid_has_four_panels() -> None:
    fig = plot_field_heatmap_grid()
    # 4 image axes + 1 colorbar axis = 5 axes total.
    assert len(fig.axes) >= 4
    # Each of the first four axes should have an image.
    for ax in fig.axes[:4]:
        assert len(ax.images) == 1
