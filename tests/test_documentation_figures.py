"""Smoke tests for the README-figure render helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from usetheforce.missions import ALL_ADAPTERS, MISSIONS, VEHICLES, evaluate_snapshot, run_mission
from usetheforce.missions.render import (
    plot_falloff_vs_distance,
    plot_mission_delta_v_bar,
    plot_twr_heatmap,
    plot_vehicle_scale_strip,
)


def test_vehicle_scale_strip_returns_figure_with_six_points() -> None:
    fig = plot_vehicle_scale_strip(VEHICLES)
    ax = fig.axes[0]
    # One line + one scatter — the scatter has six offsets.
    scatter_collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
    assert scatter_collections, "expected at least one scatter collection"
    n_points = len(scatter_collections[0].get_offsets())
    assert n_points == len(VEHICLES) == 6


def test_twr_heatmap_marks_casimir_inapplicable() -> None:
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    fig = plot_twr_heatmap(matrix, VEHICLES)
    texts = [t.get_text() for t in fig.axes[0].texts]
    # Each Casimir cell × 6 vehicles = 12 "n/a" labels.
    assert texts.count("n/a") == 12


def test_falloff_vs_distance_one_line_per_model() -> None:
    sma = VEHICLES["smallsat"]
    fields = {
        mk: ALL_ADAPTERS[mk](sma, sma.power_w).field
        for mk in [
            "shaped_field_ansatz",
            "heavy_element_lattice",
            "stimulated_emission_array",
            "antimatter_graviton",
        ]
    }
    fig = plot_falloff_vs_distance(fields, r_min_m=0.1, r_max_m=1e3, n_points=50)
    assert len(fig.axes[0].lines) == 4


def test_mission_delta_v_bar_one_bar_per_result() -> None:
    pairs = [
        ("smallsat", "shaped_field_ansatz", "free_burn_100s"),
        ("crewed", "stimulated_emission_array", "leo_raise_100s"),
    ]
    results = [
        run_mission(VEHICLES[vk], mk, ALL_ADAPTERS[mk], MISSIONS[mission])
        for vk, mk, mission in pairs
    ]
    fig = plot_mission_delta_v_bar(results)
    # Number of bars in the BarContainer should match.
    bar_containers = [c for c in fig.axes[0].containers if hasattr(c, "patches")]
    assert bar_containers
    assert len(bar_containers[0].patches) == len(results)
